import copy

import torch as th
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from marl.utils.popart import PopArt
from marl.utils.torchrl import get_explained_variance

# NOTE: step itself does not inject the state into the td / nn module, that is expected to be done by the caller


def compute_returns(rewards, dones, gamma):
    """Compute discounted reward-to-go for each timestep (flat tensor)."""
    returns = th.zeros_like(rewards)
    running_return = th.zeros(1, device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        running_return = rewards[t] + gamma * running_return * (~dones[t])
        returns[t] = running_return
    return returns


def mc_entropy(dist, num_samples=1):
    """Estimate entropy via Monte Carlo sampling."""
    try:
        return dist.entropy()
    except NotImplementedError:
        if hasattr(dist, "rsample"):
            x = dist.rsample((num_samples,))
        else:
            x = dist.sample((num_samples,))
        log_prob = dist.log_prob(x)
        return -log_prob.mean(0)


class StepTrain:
    def __init__(self, hp, num_actors):
        self.hp = hp

        self.num_actors = num_actors

        self.clear_networks()

        self.actor_order = th.randperm(num_actors)
        self.actor_order_i = 0

    def clear_networks(self):
        self.actors = None
        self.actor_optims = None

        self.pm_critic = None
        self.pm_critic_optim = None
        self.pm_popart_wrapper = None

        self.br_critics = None
        self.br_critic_optims = None
        self.br_popart_wrappers = None

        self.phi = None
        self.phi_optim = None

        # support

        self.pm_adv_mod = None
        self.pm_loss_mod = None

        self.br_adv_mods = None
        self.br_loss_mods = None

    def copy_networks(self, actors=None, pm_critic=None, br_critics=None, phi=None):
        if actors is not None:
            self.actors = copy.deepcopy(actors)
            self.actor_optims = [
                th.optim.Adam(actor.trl_actor.parameters(), lr=self.hp.lr)
                for actor in self.actors
            ]

        if pm_critic is not None:
            self.pm_critic = copy.deepcopy(pm_critic)
            self.pm_critic_optim = th.optim.Adam(
                self.pm_critic.parameters(), lr=self.hp.lr
            )

            if self.hp.popart_pm:
                self.pm_popart_wrapper = PopArt(
                    self.pm_critic, beta=self.hp.popart_beta
                )
            else:
                self.pm_popart_wrapper = None

        if br_critics is not None:
            self.br_critics = copy.deepcopy(br_critics)
            self.br_critic_optims = [
                th.optim.Adam(br_critic.parameters(), lr=self.hp.lr)
                for br_critic in self.br_critics
            ]

            if self.hp.popart:
                self.br_popart_wrappers = [
                    PopArt(br_critic, beta=self.hp.popart_beta)
                    for br_critic in self.br_critics
                ]
            else:
                self.br_popart_wrappers = None

        if phi is not None:
            # Check if phi is an ensemble wrapper - store reference instead of deepcopy
            # The wrapper class is defined in neppo.py and has phi_ensemble attribute
            if hasattr(phi, "phi_ensemble"):
                # This is an EnsemblePhiWrapper - store reference directly
                self.phi = phi
                self.phi_optim = None  # Ensemble manages its own optimizers
            else:
                self.phi = copy.deepcopy(phi)
                self.phi_optim = th.optim.Adam(self.phi.parameters(), lr=self.hp.phi_lr)

        self.make_params_contiguous()

        # support

        if actors is not None or pm_critic is not None:
            self.pm_loss_mod = [
                ClipPPOLoss(
                    actor_network=actor.trl_actor,
                    critic_network=self.pm_critic,
                    clip_epsilon=self.hp.clip_epsilon,
                    entropy_bonus=bool(self.hp.entropy_eps),
                    entropy_coeff=self.hp.entropy_eps,
                    critic_coeff=self.hp.critic_coef,
                    loss_critic_type=self.hp.loss_critic_type,
                )
                for actor in self.actors
            ]

        if pm_critic is not None:
            pm_value_network = (
                self.pm_popart_wrapper if self.hp.popart_pm else self.pm_critic
            )
            self.pm_adv_mod = GAE(
                gamma=self.hp.gamma,
                lmbda=self.hp.gae_lambda,
                value_network=pm_value_network,
                average_gae=self.hp.normalize_advantage,
                deactivate_vmap=True,
            )

        if br_critics is not None:
            br_value_networks = (
                self.br_popart_wrappers if self.hp.popart else self.br_critics
            )
            self.br_adv_mods = [
                GAE(
                    gamma=self.hp.gamma,
                    lmbda=self.hp.gae_lambda,
                    value_network=br_value_net,
                    average_gae=self.hp.normalize_advantage,
                    deactivate_vmap=True,
                )
                for br_value_net in br_value_networks
            ]
        if actors is not None or br_critics is not None:
            self.br_loss_mods = [
                ClipPPOLoss(
                    actor_network=actor.trl_actor,
                    critic_network=br_critic,
                    clip_epsilon=self.hp.clip_epsilon,
                    entropy_bonus=bool(self.hp.entropy_eps),
                    entropy_coeff=self.hp.entropy_eps,
                )
                for actor, br_critic in zip(self.actors, self.br_critics)
            ]

    def rebuild_modules(self):
        """Rebuild loss/advantage modules to restore parameter sharing after deepcopy.

        TorchRL's LossModule may store parameters in a functional representation
        that doesn't survive deepcopy — the loss module ends up with its own
        detached copies of the actor/critic params. Rebuilding from scratch
        ensures the modules reference the actual self.actors / self.pm_critic /
        self.br_critics parameters.
        """
        self.pm_loss_mod = [
            ClipPPOLoss(
                actor_network=actor.trl_actor,
                critic_network=self.pm_critic,
                clip_epsilon=self.hp.clip_epsilon,
                entropy_bonus=bool(self.hp.entropy_eps),
                entropy_coeff=self.hp.entropy_eps,
                critic_coeff=self.hp.critic_coef,
                loss_critic_type=self.hp.loss_critic_type,
            )
            for actor in self.actors
        ]

        if self.hp.popart_pm:
            self.pm_popart_wrapper = PopArt(self.pm_critic, beta=self.hp.popart_beta)
        else:
            self.pm_popart_wrapper = None

        pm_value_network = (
            self.pm_popart_wrapper if self.hp.popart_pm else self.pm_critic
        )
        self.pm_adv_mod = GAE(
            gamma=self.hp.gamma,
            lmbda=self.hp.gae_lambda,
            value_network=pm_value_network,
            average_gae=self.hp.normalize_advantage,
            deactivate_vmap=True,
        )

        if self.hp.popart:
            self.br_popart_wrappers = [
                PopArt(br_critic, beta=self.hp.popart_beta)
                for br_critic in self.br_critics
            ]
        else:
            self.br_popart_wrappers = None

        br_value_networks = (
            self.br_popart_wrappers if self.hp.popart else self.br_critics
        )
        self.br_adv_mods = [
            GAE(
                gamma=self.hp.gamma,
                lmbda=self.hp.gae_lambda,
                value_network=br_value_net,
                average_gae=self.hp.normalize_advantage,
                deactivate_vmap=True,
            )
            for br_value_net in br_value_networks
        ]

        self.br_loss_mods = [
            ClipPPOLoss(
                actor_network=actor.trl_actor,
                critic_network=br_critic,
                clip_epsilon=self.hp.clip_epsilon,
                entropy_bonus=bool(self.hp.entropy_eps),
                entropy_coeff=self.hp.entropy_eps,
            )
            for actor, br_critic in zip(self.actors, self.br_critics)
        ]

        self.actor_optims = [
            th.optim.Adam(actor.trl_actor.parameters(), lr=self.hp.lr)
            for actor in self.actors
        ]

        self.pm_critic_optim = th.optim.Adam(self.pm_critic.parameters(), lr=self.hp.lr)

        self.br_critic_optims = [
            th.optim.Adam(br_critic.parameters(), lr=self.hp.lr)
            for br_critic in self.br_critics
        ]

        # Only recreate phi optimizer if not using ensemble wrapper
        if self.phi is not None and not hasattr(self.phi, "phi_ensemble"):
            self.phi_optim = th.optim.Adam(self.phi.parameters(), lr=self.hp.phi_lr)

    def make_params_contiguous(self):
        for actor in self.actors:
            for param in actor.trl_actor.parameters():
                param.data = param.data.contiguous()
        for param in self.pm_critic.parameters():
            param.data = param.data.contiguous()
        for critic in self.br_critics:
            for param in critic.parameters():
                param.data = param.data.contiguous()
        # Skip ensemble wrapper - it manages its own parameters
        if self.phi is not None and not hasattr(self.phi, "phi_ensemble"):
            for param in self.phi.parameters():
                param.data = param.data.contiguous()

    def pm(self, tds, log_prefix="pm"):
        for i in range(self.num_actors):
            tds[i]["action_log_prob"] = (
                self.actors[i]
                .trl_actor.get_dist(tds[i])
                .log_prob(tds[i]["action"])
                .detach()
            )

        rewards = self.phi(
            tds[0]["state"],
            th.stack([tds[j]["next", "reward"] for j in range(self.num_actors)]),
        )

        for i in range(self.num_actors):
            tds[i]["next", "reward"] = rewards.detach()

        # Advantage computation: GRPO vs GAE
        if self.hp.grpo_pm:
            # GRPO: per-batch normalized reward-to-go (no critic/GAE)
            grpo_rewards = tds[0]["next", "reward"]
            grpo_dones = tds[0]["next", "done"]
            returns = compute_returns(grpo_rewards, grpo_dones, self.hp.gamma)
            grpo_advantage = (returns - returns.mean()) / (returns.std() + 1e-8)
            for i in range(self.num_actors):
                tds[i]["advantage"] = grpo_advantage.detach()
        else:
            for i in range(self.num_actors):
                tds[i] = self.pm_adv_mod(tds[i])

        # Update PopArt only when not using GRPO
        if (
            not self.hp.grpo_pm
            and self.hp.popart_pm
            and self.pm_popart_wrapper is not None
        ):
            self.pm_popart_wrapper.update(tds[0]["value_target"])
            for i in range(self.num_actors):
                tds[i]["value_target"] = self.pm_popart_wrapper.normalize_target(
                    tds[i]["value_target"]
                )

        log_dict = {}

        kappa = 1
        while self.actor_order_i < self.num_actors:
            i = self.actor_order[self.actor_order_i]
            for _ in range(self.hp.num_epochs):
                td = copy.deepcopy(tds[i])
                td["advantage"] *= kappa

                if self.hp.grpo_pm:
                    # GRPO: compute PPO loss manually
                    dist = self.actors[i].trl_actor.get_dist(td)
                    log_prob = dist.log_prob(td["action"])
                    ratio = th.exp(log_prob - td["action_log_prob"])
                    adv = td["advantage"]
                    surr1 = ratio * adv
                    surr2 = (
                        th.clamp(
                            ratio, 1 - self.hp.clip_epsilon, 1 + self.hp.clip_epsilon
                        )
                        * adv
                    )
                    loss_objective = -th.min(surr1, surr2).mean()
                    loss_entropy = (
                        -mc_entropy(dist).mean() * self.hp.entropy_eps
                        if self.hp.entropy_eps
                        else th.tensor(0.0)
                    )
                    loss_value = loss_objective + loss_entropy
                    loss_vals = {
                        "loss_objective": loss_objective,
                        "loss_entropy": loss_entropy,
                    }
                else:
                    loss_vals = self.pm_loss_mod[i](td)
                    loss_value = loss_vals["loss_objective"] + (
                        loss_vals["loss_entropy"] if self.hp.entropy_eps else 0
                    )

                self.actor_optims[i].zero_grad()
                loss_value.backward()
                loss_vals["loss_grad_norm"] = th.nn.utils.clip_grad_norm_(
                    self.actors[i].trl_actor.parameters(),
                    (
                        self.hp.actor_max_grad_norm
                        if self.hp.actor_clip_grad_norm
                        else float("inf")
                    ),
                )
                self.actor_optims[i].step()

                # logging
                if "loss_critic" in loss_vals:
                    del loss_vals["loss_critic"]
                for k, v in loss_vals.items():
                    log_dict[f"{log_prefix}/train/{i}/{k}"] = v.item()

            kappa *= th.exp(
                self.actors[i]
                .trl_actor.get_dist(tds[i])
                .log_prob(tds[i]["action"])
                .detach()
                - td["action_log_prob"]
            )[..., 0]

            self.actor_order_i += 1
        self.actor_order_i = 0

        self.actor_order = th.randperm(self.num_actors)

        # critic (skip when using GRPO)
        if not self.hp.grpo_pm:
            i = 0
            for _ in range(self.hp.num_epochs):
                loss_vals = self.pm_loss_mod[i](tds[i])
                self.pm_critic_optim.zero_grad()
                loss_vals["loss_critic"].backward()
                critic_grad_norm = th.nn.utils.clip_grad_norm_(
                    self.pm_critic.parameters(),
                    (
                        self.hp.critic_max_grad_norm
                        if self.hp.critic_clip_grad_norm
                        else float("inf")
                    ),
                )
                self.pm_critic_optim.step()
            log_dict[f"{log_prefix}/train/loss_critic"] = loss_vals[
                "loss_critic"
            ].item()
            log_dict[f"{log_prefix}/train/critic_grad_norm"] = critic_grad_norm.item()

        # logging
        for i in range(self.num_actors):
            dist = self.actors[i].trl_actor.get_dist(tds[i])
            if hasattr(dist, "scale"):
                log_dict[f"{log_prefix}/train/{i}/action_std"] = (
                    dist.scale.mean().item()
                )
            else:
                log_dict[f"{log_prefix}/train/{i}/action_std"] = 0.0

        if not self.hp.grpo_pm:
            log_dict[f"{log_prefix}/train/explained_variance"] = get_explained_variance(
                tds[i]["state_value"], tds[i]["value_target"]
            )

        # Compute phi if needed
        log_dict[f"{log_prefix}/train/avg_phi"] = rewards.detach().mean().item()
        num_done = tds[i]["next", "done"].sum().item()
        if num_done > 0:
            log_dict[f"{log_prefix}/train/avg_sum_phi"] = (
                rewards.detach().sum().item() / num_done
            )
            log_dict[f"{log_prefix}/train/avg_Phi"] = (
                rewards * th.pow(self.hp.gamma, tds[i]["step_count"])
            ).sum().item() / num_done

        # PopArt logging
        if self.hp.popart_pm and self.pm_popart_wrapper is not None:
            log_dict[f"{log_prefix}/train/popart_mu"] = self.pm_popart_wrapper.mu.item()
            log_dict[f"{log_prefix}/train/popart_sigma"] = (
                self.pm_popart_wrapper.sigma.item()
            )

        return log_dict, True, tds[i].shape[0]

    def br(self, tds, log_prefix="br"):
        i = self.actor_order[self.actor_order_i]
        td = tds[i]

        td["action_log_prob"] = (
            self.actors[i].trl_actor.get_dist(td).log_prob(td["action"]).detach()
        )
        td = self.br_adv_mods[i](td)

        # Update PopArt and normalize targets BEFORE training
        if self.hp.popart and self.br_popart_wrappers is not None:
            self.br_popart_wrappers[i].update(td["value_target"])
            td["value_target"] = self.br_popart_wrappers[i].normalize_target(
                td["value_target"]
            )

        for _ in range(self.hp.num_epochs):
            loss_vals = self.br_loss_mods[i](td)
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + (loss_vals["loss_entropy"] if self.hp.entropy_eps else 0)
            )

            self.actor_optims[i].zero_grad()
            self.br_critic_optims[i].zero_grad()

            loss_value.backward()

            # Clip actor gradient individually
            actor_grad_norm = th.nn.utils.clip_grad_norm_(
                self.actors[i].trl_actor.parameters(),
                (
                    self.hp.actor_max_grad_norm
                    if self.hp.actor_clip_grad_norm
                    else float("inf")
                ),
            )

            # Clip critic gradient individually
            critic_grad_norm = th.nn.utils.clip_grad_norm_(
                self.br_critics[i].parameters(),
                (
                    self.hp.critic_max_grad_norm
                    if self.hp.critic_clip_grad_norm
                    else float("inf")
                ),
            )

            self.actor_optims[i].step()
            self.br_critic_optims[i].step()

        log_dict = {}
        for k, v in loss_vals.items():
            log_dict[f"{log_prefix}/t{i}/train/{i}/{k}"] = v.item()
        log_dict[f"{log_prefix}/t{i}/train/{i}/actor_grad_norm"] = (
            actor_grad_norm.item()
        )
        log_dict[f"{log_prefix}/t{i}/train/{i}/critic_grad_norm"] = (
            critic_grad_norm.item()
        )
        log_dict[f"{log_prefix}/t{i}/train/{i}/explained_variance"] = (
            get_explained_variance(td["state_value"], td["value_target"])
        )
        dist = self.actors[i].trl_actor.get_dist(td)
        if hasattr(dist, "scale"):
            log_dict[f"{log_prefix}/t{i}/train/{i}/action_std"] = (
                dist.scale.mean().item()
            )
        else:
            log_dict[f"{log_prefix}/t{i}/train/{i}/action_std"] = 0.0
        log_dict[f"{log_prefix}/rollout/agent_idx"] = i

        rewards = self.phi(
            td["state"],
            th.stack([tds[j]["next", "reward"] for j in range(self.num_actors)]),
        )

        num_done = td["next", "done"].sum().item()
        if num_done > 0:
            log_dict[f"{log_prefix}/t{i}/train/avg_Phi"] = (
                rewards * th.pow(self.hp.gamma, td["step_count"])
            ).sum().item() / num_done
        log_dict[f"{log_prefix}/t{i}/train/avg_phi"] = rewards.detach().mean().item()

        # PopArt logging
        if self.hp.popart and self.br_popart_wrappers is not None:
            log_dict[f"{log_prefix}/t{i}/train/{i}/popart_mu"] = (
                self.br_popart_wrappers[i].mu.item()
            )
            log_dict[f"{log_prefix}/t{i}/train/{i}/popart_sigma"] = (
                self.br_popart_wrappers[i].sigma.item()
            )

        self.actor_order_i += 1
        iter_flag = False
        if self.actor_order_i == self.num_actors:
            iter_flag = True

            self.actor_order = th.randperm(self.num_actors)
            self.actor_order_i = 0

        return log_dict, iter_flag, td.shape[0]
