import copy
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn.functional as F

from marl.algorithms.neppo.potdiff import PotDiffStep
from marl.algorithms.neppo.step import StepTrain


@dataclass
class Hyperparameters:
    pm_zo_steps: int = 1
    br_zo_steps: int = 1

    phi_lr: float = 1e-4
    phi_epochs: int = 1
    phi_max_grad_norm: float = 10.0
    phi_clip_grad_norm: bool = True

    actor_max_grad_norm: float = 10.0
    actor_clip_grad_norm: bool = True
    critic_max_grad_norm: float = 10.0
    critic_clip_grad_norm: bool = True

    zeroth_order_delta: float = 1e-3
    # lse_beta: float = 1.0  # TODO: implement in softmax / lse

    mixer: bool = False
    mixer_input_dim: int = 1
    mixer_output_dim: int = 1

    normalize_advantage: bool = True

    # ppo params
    entropy_eps: float = 5e-3
    num_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    critic_coef: float = 0.5
    lr: float = 3e-4
    loss_critic_type: str = "smooth_l1"

    # PopArt value normalization
    popart: bool = False  # For BR training
    popart_pm: bool = False  # For PM training
    popart_beta: float = 0.0001

    # GRPO loss for PM step (no critic)
    grpo_pm: bool = True
    adam_phi: bool = False
    nonmix: bool = False
    nonmix_hidden_size: int = 64
    convex_nn: bool = False
    convex_nn_hidden_size: int = 16
    mixer_with_delta: bool = False

    phi_polyak: bool = False
    phi_polyak_tau: float = 0.005
    phi_lr_inverse_sqrt: bool = False

    ensemble_disagreement: bool = False
    ensemble_disagreement_N: int = 5
    ensemble_disagreement_alpha: float = 0.1


class EnsemblePhiWrapper:
    """Wrapper that provides same interface as single phi but uses ensemble with variance penalty."""

    def __init__(self, phi_ensemble, alpha):
        self.phi_ensemble = phi_ensemble
        self.alpha = alpha

    def __call__(self, state, reward):
        outputs = th.stack([phi_i(state, reward) for phi_i in self.phi_ensemble])
        mean_output = outputs.mean(dim=0)
        var_output = outputs.var(dim=0, unbiased=False)
        return mean_output - self.alpha * var_output

    def parameters(self):
        # Return parameters from first phi for contiguity checks (all have same structure)
        return self.phi_ensemble[0].parameters()

    def named_parameters(self):
        return self.phi_ensemble[0].named_parameters()


def get_trajectory_indices(done):
    """
    Given flattened done flags, return list of index arrays, one per trajectory.
    done: [total_timesteps] boolean tensor
    Returns: list of [traj_len] index tensors
    """
    # Find indices where done=True (trajectory boundaries)
    done_indices = th.where(done)[0]

    trajectories = []
    start = 0
    for end_idx in done_indices:
        # Trajectory includes timesteps from start to end_idx (inclusive)
        traj_indices = th.arange(start, end_idx + 1, device=done.device)
        trajectories.append(traj_indices)
        start = end_idx + 1

    # Handle any remaining timesteps after last done (incomplete trajectory)
    if start < len(done):
        trajectories.append(th.arange(start, len(done), device=done.device))

    return trajectories


def bootstrap_trajectories(td, done_key="next/done"):
    """
    Bootstrap sample trajectories with replacement.
    Returns indices to reconstruct a bootstrapped dataset of same size.
    """
    done = td[done_key].squeeze()
    trajectories = get_trajectory_indices(done)
    num_traj = len(trajectories)

    # Sample trajectory indices with replacement
    sampled_traj_ids = np.random.choice(num_traj, size=num_traj, replace=True)

    # Gather timestep indices from sampled trajectories
    bootstrap_indices = th.cat([trajectories[i] for i in sampled_traj_ids])

    return bootstrap_indices


class NePPO:
    def __init__(self, actors, pm_critic, br_critics, phi, device, **hp):
        self.hp = Hyperparameters(**hp)
        self.num_actors = len(actors)

        # Validate mutual exclusivity
        if self.hp.phi_polyak and self.hp.ensemble_disagreement:
            raise ValueError(
                "Cannot use both phi_polyak and ensemble_disagreement simultaneously"
            )

        self.phi = phi

        if self.hp.ensemble_disagreement:
            # Create N copies of phi for ensemble
            self.phi_ensemble = [
                copy.deepcopy(phi) for _ in range(self.hp.ensemble_disagreement_N)
            ]
            self.phi_optims = [
                th.optim.SGD(phi_i.parameters(), lr=self.hp.phi_lr)
                for phi_i in self.phi_ensemble
            ]
            # Create wrapper that provides same interface as single phi
            self.phi_wrapper = EnsemblePhiWrapper(
                self.phi_ensemble, self.hp.ensemble_disagreement_alpha
            )
            self.phi_optim = None  # Not used for ensemble
        else:
            self.phi_ensemble = None
            self.phi_optims = None
            self.phi_wrapper = None
            self.phi_optim = th.optim.SGD(self.phi.parameters(), lr=self.hp.phi_lr)

        if self.hp.phi_polyak:
            self.phi_target = copy.deepcopy(self.phi)
            for param in self.phi_target.parameters():
                param.requires_grad = False
        else:
            self.phi_target = None

        self.pm = StepTrain(self.hp, self.num_actors)
        self.br = StepTrain(self.hp, self.num_actors)

        self.hat_pm = None
        self.hat_br = None

        self.pd = PotDiffStep(self.hp, self.num_actors)
        self.hat_pd = PotDiffStep(self.hp, self.num_actors)

        if self.hp.ensemble_disagreement:
            phi_for_training = self.phi_wrapper
        elif self.hp.phi_polyak:
            phi_for_training = self.phi_target
        else:
            phi_for_training = phi
        self.pm.copy_networks(
            actors=actors,
            pm_critic=pm_critic,
            br_critics=br_critics,
            phi=phi_for_training,
        )
        self.br.copy_networks(
            actors=actors,
            pm_critic=pm_critic,
            br_critics=br_critics,
            phi=phi_for_training,
        )

        self.u = {}

        self.branch = 0
        self.iter = 0
        self.phi_update_count = 0

        self.device = device

        self.phi_num_params = sum(p.numel() for p in self.phi.parameters())

    def _get_phi_lr(self):
        if not self.hp.phi_lr_inverse_sqrt:
            return self.hp.phi_lr
        step = self.phi_update_count + 1  # 1-indexed to avoid division by zero
        return self.hp.phi_lr / (step**0.5)

    @property
    def current_policies(self):
        match self.branch:
            case 0:  # potmax
                return self.pm.actors
            case 1:  # potmax hat
                return self.hat_pm.actors
            case 2:  # br
                actors = [actor for actor in self.pm.actors]
                i = self.br.actor_order[self.br.actor_order_i]
                actors[i] = self.br.actors[i]
                return actors
            case 3:  # br hat
                actors = [actor for actor in self.hat_pm.actors]
                i = self.hat_br.actor_order[self.hat_br.actor_order_i]
                actors[i] = self.hat_br.actors[i]
                return actors
            case 4:  # potdiff
                return self.pd.actors
            case 5:  # potdiff hat
                return self.hat_pd.actors

    def step(self, tds):
        _log_dict = {
            "debug/branch": self.branch,
            "debug/iter": self.iter,
        }

        match self.branch:
            case 0:  # potmax
                log_dict, iter_flag, _ = self.pm.pm(tds, log_prefix="pm")
                self.iter += int(iter_flag)
                if self.iter >= self.hp.pm_zo_steps:
                    self.branch = 1
                    self.iter = 0

                    self.hat_pm = copy.deepcopy(self.pm)
                    self.hat_pm.rebuild_modules()

                    hat_phi, check_phi = self.sample_zo_phi()

                    self.hat_pm.copy_networks(phi=hat_phi)
                    self.hat_pm.make_params_contiguous()

                    self.pm.copy_networks(phi=check_phi)
                    self.pm.make_params_contiguous()

                    # for logging
                    if self.hp.ensemble_disagreement:
                        self.br.copy_networks(phi=self.phi_wrapper)
                    elif self.hp.phi_polyak:
                        self.br.copy_networks(phi=self.phi_target)
                    else:
                        self.br.copy_networks(phi=self.phi)

            case 1:  # potmax hat
                log_dict, iter_flag, _ = self.hat_pm.pm(tds, log_prefix="hat_pm")
                self.iter += int(iter_flag)
                if self.iter >= self.hp.pm_zo_steps:
                    self.branch = 2
                    self.iter = 0

            case 2:  # br
                log_dict, iter_flag, _ = self.br.br(tds, "br")
                self.iter += int(iter_flag)
                if self.iter >= self.hp.br_zo_steps:
                    self.branch = 3
                    self.iter = 0

                    self.hat_br = copy.deepcopy(self.br)
                    self.hat_br.rebuild_modules()
                    self.hat_br.make_params_contiguous()

            case 3:  # br hat
                log_dict, iter_flag, _ = self.hat_br.br(tds, "hat_br")
                self.iter += int(iter_flag)
                if self.iter >= self.hp.br_zo_steps:
                    self.branch = 4
                    self.iter = 0

                    self.pd.set_networks(self.pm.actors, self.br.actors, self.pm.phi)
                    self.hat_pd.set_networks(
                        self.hat_pm.actors, self.hat_br.actors, self.hat_pm.phi
                    )

            case 4:  # potdiff
                log_dict, iter_flag, _ = self.pd.save_dataset(tds, log_prefix="pd")
                if iter_flag:
                    self.branch = 5

            case 5:  # potdiff hat
                log_dict, iter_flag, _ = self.hat_pd.save_dataset(
                    tds, log_prefix="hat_pd"
                )
                if iter_flag:
                    log_dict |= self.phi_update("pu")
                    self.pd.clear_datasets()
                    self.hat_pd.clear_datasets()

                    self.branch = 0
                    self.iter = 0

                    self.u = {}
                    self.hat_pm = None
                    self.hat_br = None

                    self.pd.clear_networks()
                    self.hat_pd.clear_networks()

        log_dict |= _log_dict

        return log_dict

    def _compute_fs_with_phi(self, phi, pm_dataset, br_datasets):
        """Compute F values using a specific phi network."""
        fs = []
        for i in range(self.num_actors):
            td = pm_dataset[i]
            br_td = br_datasets[i]

            pm_rewards = phi(td["state"], td["rewards_T"].T)
            br_rewards = phi(br_td["state"], br_td["rewards_T"].T)

            br_Phi = (1 / br_td["next", "done"].sum()) * (
                th.sum(br_rewards * th.pow(self.hp.gamma, br_td["step_count"]))
            )
            Phi = (1 / td["next", "done"].sum()) * (
                th.sum(pm_rewards * th.pow(self.hp.gamma, td["step_count"]))
            )

            br_V = (1 / br_td["next", "done"].sum()) * (
                th.sum(
                    br_td["next", "reward"] * th.pow(self.hp.gamma, br_td["step_count"])
                )
            )
            V = (1 / td["next", "done"].sum()) * (
                th.sum(td["next", "reward"] * th.pow(self.hp.gamma, td["step_count"]))
            )

            fs.append((Phi - br_Phi) - (V - br_V))
        return fs

    def _bootstrap_dataset(self, pm_dataset, br_datasets):
        """Bootstrap sample trajectories with replacement for both pm and br datasets."""
        bootstrapped_pm = []
        bootstrapped_br = []

        for i in range(self.num_actors):
            # Bootstrap pm dataset for agent i
            pm_indices = bootstrap_trajectories(pm_dataset[i])
            bootstrapped_pm.append(pm_dataset[i][pm_indices])

            # Bootstrap br dataset for agent i
            br_indices = bootstrap_trajectories(br_datasets[i])
            bootstrapped_br.append(br_datasets[i][br_indices])

        return bootstrapped_pm, bootstrapped_br

    def phi_update(self, log_prefix="pu"):
        dF_dw_tot_l2_norm = 0

        for e in range(self.hp.phi_epochs):
            fs, pd_log_dict = self.pd.fs()
            hat_fs, hat_pd_log_dict = self.hat_pd.fs()

            st_fs = th.stack(fs)
            st_hat_fs = th.stack(hat_fs)

            lse_fs = th.logsumexp(st_fs, dim=0).detach()
            lse_hat_fs = th.logsumexp(st_hat_fs, dim=0).detach()

            if self.hp.ensemble_disagreement:
                # Train each ensemble member with bootstrapped data
                ensemble_grad_norms = []
                for m in range(self.hp.ensemble_disagreement_N):
                    # Bootstrap right before gradient computation for this phi_m
                    pm_boot, br_boot = self._bootstrap_dataset(
                        self.pd.pm_dataset, self.pd.br_datasets
                    )
                    hat_pm_boot, hat_br_boot = self._bootstrap_dataset(
                        self.hat_pd.pm_dataset, self.hat_pd.br_datasets
                    )

                    # Compute fs using bootstrapped data for this phi_m
                    fs_m = self._compute_fs_with_phi(
                        self.phi_ensemble[m], pm_boot, br_boot
                    )
                    hat_fs_m = self._compute_fs_with_phi(
                        self.phi_ensemble[m], hat_pm_boot, hat_br_boot
                    )

                    st_fs_m = th.stack(fs_m)
                    st_hat_fs_m = th.stack(hat_fs_m)

                    lse_fs_m = th.logsumexp(st_fs_m, dim=0).detach()
                    lse_hat_fs_m = th.logsumexp(st_hat_fs_m, dim=0).detach()

                    self.phi_optims[m].zero_grad()

                    with th.no_grad():
                        for name, param in self.phi_ensemble[m].named_parameters():
                            param.grad = (
                                self.phi_num_params
                                / (2 * self.hp.zeroth_order_delta)
                                * (lse_hat_fs_m - lse_fs_m)
                                * self.u[name]
                            )

                    grad_norm = th.nn.utils.clip_grad_norm_(
                        self.phi_ensemble[m].parameters(),
                        (
                            self.hp.phi_max_grad_norm
                            if self.hp.phi_clip_grad_norm
                            else float("inf")
                        ),
                    )
                    ensemble_grad_norms.append(grad_norm)

                    new_lr = self._get_phi_lr()
                    for param_group in self.phi_optims[m].param_groups:
                        param_group["lr"] = new_lr
                    self.phi_optims[m].step()

                # Use mean grad norm for logging
                dF_dw_tot_l2_norm = th.stack(ensemble_grad_norms).mean()
            else:
                self.phi_optim.zero_grad()

                with th.no_grad():
                    for i in range(self.num_actors):
                        for name, param in self.phi.named_parameters():
                            param.grad = (
                                self.phi_num_params
                                / (2 * self.hp.zeroth_order_delta)
                                * (lse_hat_fs - lse_fs)
                                * self.u[name]
                            )

                dF_dw_tot_l2_norm = th.nn.utils.clip_grad_norm_(
                    self.phi.parameters(),
                    (
                        self.hp.phi_max_grad_norm
                        if self.hp.phi_clip_grad_norm
                        else float("inf")
                    ),
                )

                new_lr = self._get_phi_lr()
                for param_group in self.phi_optim.param_groups:
                    param_group["lr"] = new_lr
                self.phi_optim.step()

                # Polyak averaging update
                if self.hp.phi_polyak and self.phi_target is not None:
                    with th.no_grad():
                        for param, target_param in zip(
                            self.phi.parameters(), self.phi_target.parameters()
                        ):
                            target_param.data.copy_(
                                self.hp.phi_polyak_tau * param.data
                                + (1 - self.hp.phi_polyak_tau) * target_param.data
                            )

                # Print output layer z after phi iteration when convex_nn is on
                if self.hp.convex_nn:
                    print(
                        f"[phi iter {e}] output_layer_z weight:\n{self.phi.output_layer_z.weight}"
                    )

        self.phi_update_count += 1

        # logging
        log_dict = {}
        log_dict[f"{log_prefix}/phi_lr"] = self._get_phi_lr()

        log_dict |= {f"pd/{k}": v for k, v in pd_log_dict.items()}
        log_dict |= {f"hat_pd/{k}": v for k, v in hat_pd_log_dict.items()}

        log_dict[f"{log_prefix}/train/dF_dw_l2_norm"] = dF_dw_tot_l2_norm.item()

        max_i = 0

        for i in range(self.num_actors):
            if abs(fs[i]) > abs(fs[max_i]):
                max_i = i
            log_dict[f"{log_prefix}/train/{i}/F_i"] = fs[i].item()
            log_dict[f"{log_prefix}/train/{i}/hat_F_i"] = hat_fs[i].item()
            log_dict[f"{log_prefix}/train/{i}/abs_F_i"] = abs(fs[i]).item()

        log_dict[f"{log_prefix}/train/max_F_i"] = fs[max_i].item()
        log_dict[f"{log_prefix}/train/max_abs_F_i"] = abs(fs[max_i]).item()
        log_dict[f"{log_prefix}/train/max_i"] = max_i
        log_dict[f"{log_prefix}/train/lse"] = th.logsumexp(st_fs, dim=0).detach().item()

        # Log phi_target LSE for monitoring (if Polyak averaging enabled)
        if self.hp.phi_polyak and self.phi_target is not None:
            with th.no_grad():
                target_fs = []
                for i in range(self.num_actors):
                    td = self.pd.pm_dataset[i]
                    br_td = self.pd.br_datasets[i]

                    pm_rewards = self.phi_target(td["state"], td["rewards_T"].T)
                    br_rewards = self.phi_target(br_td["state"], br_td["rewards_T"].T)

                    br_Phi = (1 / br_td["next", "done"].sum()) * (
                        th.sum(br_rewards * th.pow(self.hp.gamma, br_td["step_count"]))
                    )
                    Phi = (1 / td["next", "done"].sum()) * (
                        th.sum(pm_rewards * th.pow(self.hp.gamma, td["step_count"]))
                    )

                    # V and br_V from individual rewards (same as potdiff.fs())
                    br_V = (1 / br_td["next", "done"].sum()) * (
                        th.sum(
                            br_td["next", "reward"]
                            * th.pow(self.hp.gamma, br_td["step_count"])
                        )
                    )
                    V = (1 / td["next", "done"].sum()) * (
                        th.sum(
                            td["next", "reward"]
                            * th.pow(self.hp.gamma, td["step_count"])
                        )
                    )

                    target_fs.append((Phi - br_Phi) - (V - br_V))

                st_target_fs = th.stack(target_fs)
                log_dict[f"{log_prefix}/train/target_lse"] = th.logsumexp(
                    st_target_fs, dim=0
                ).item()

        # Log ensemble variance for monitoring (if ensemble disagreement enabled)
        if self.hp.ensemble_disagreement and self.phi_ensemble is not None:
            with th.no_grad():
                # Compute phi outputs for each ensemble member
                td = self.pd.pm_dataset[0]
                state = td["state"]
                rewards_T = td["rewards_T"].T
                ensemble_outputs = th.stack(
                    [phi_i(state, rewards_T) for phi_i in self.phi_ensemble]
                )
                # Log mean variance across timesteps
                var_output = ensemble_outputs.var(dim=0, unbiased=False)
                log_dict[f"{log_prefix}/train/ensemble_var_mean"] = (
                    var_output.mean().item()
                )
                log_dict[f"{log_prefix}/train/ensemble_var_max"] = (
                    var_output.max().item()
                )

        return log_dict

    def sample_zo_phi(self):
        if self.hp.ensemble_disagreement:
            # For ensemble, perturb all members with the same u
            hat_phi_ensemble = [copy.deepcopy(phi_i) for phi_i in self.phi_ensemble]
            check_phi_ensemble = [copy.deepcopy(phi_i) for phi_i in self.phi_ensemble]

            # Generate perturbation based on first ensemble member (all have same structure)
            total_sq = 0
            with th.no_grad():
                for name, param in hat_phi_ensemble[0].named_parameters():
                    self.u[name] = th.randn_like(param).to(self.device)
                    total_sq += th.sum(th.square(self.u[name]))
                scale = th.sqrt(1 / total_sq)

                # Apply same perturbation to all ensemble members
                for name, _ in hat_phi_ensemble[0].named_parameters():
                    self.u[name] = scale * self.u[name]

                for m in range(self.hp.ensemble_disagreement_N):
                    for name, param in hat_phi_ensemble[m].named_parameters():
                        param.data = (
                            param.data + self.hp.zeroth_order_delta * self.u[name]
                        )
                    for name, param in check_phi_ensemble[m].named_parameters():
                        param.data = (
                            param.data - self.hp.zeroth_order_delta * self.u[name]
                        )

            # Wrap in EnsemblePhiWrapper
            hat_phi = EnsemblePhiWrapper(
                hat_phi_ensemble, self.hp.ensemble_disagreement_alpha
            )
            check_phi = EnsemblePhiWrapper(
                check_phi_ensemble, self.hp.ensemble_disagreement_alpha
            )

            return hat_phi, check_phi
        else:
            hat_phi = copy.deepcopy(self.phi)
            check_phi = copy.deepcopy(self.phi)

            total_sq = 0
            with th.no_grad():
                for name, param in hat_phi.named_parameters():
                    self.u[name] = th.randn_like(param).to(self.device)
                    total_sq += th.sum(th.square(self.u[name]))
                scale = th.sqrt(1 / total_sq)
                for name, param in hat_phi.named_parameters():
                    self.u[name] = scale * self.u[name]
                    param.data = param.data + self.hp.zeroth_order_delta * self.u[name]
                for name, param in check_phi.named_parameters():
                    param.data = param.data - self.hp.zeroth_order_delta * self.u[name]

            return hat_phi, check_phi
