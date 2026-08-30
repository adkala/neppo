import copy
import os
import time

import jax
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from jaxtrl.rl.ppo.actor.mlp import MLPActor
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator
from jaxtrl.utils.logger import ExtWandbLogger
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

from mpe_env import JaxMARLSim, JaxMARLStateAgent, make_jaxmarl_env
from marl.utils.collect import tdl, transitions_to_tensordicts
from marl.utils.torchrl import save_actor

DEFAULT_NAME = "jaxmarl_maddpg"
JOB_TYPE = "jaxmarl_maddpg"
SEED = None

TIMESTEPS_PER_ENV = 25
NUM_ENVS = 128

CONFIG = {
    "trainer": {
        "actor": "mlp",
        "actor_kwargs": {
            "net_arch": [64, 64],
            "activation_func": "relu",
            "distribution": "tanh",
            "distribution_kwargs": {"tanh_loc": True},
        },
        "algo_kwargs": {
            "lr_actor": 1e-2,
            "lr_critic": 1e-2,
            "gamma": 0.95,
            "tau": 0.01,
            "buffer_size": 1_000_000,
            "batch_size": 1024,
            "warmup_steps": 1024,
            "critic_hidden_dims": [64, 64],
            "grad_norm_clip": 0.5,
        },
    },
    "collect": {
        "num_envs": NUM_ENVS,
    },
    "save": {
        "freq": 1_000_000,
    },
    "env": {
        "name": "MPE_simple_world_comm_v3",
        "max_steps": 25,
    },
    "seed": None,
    "device": "cuda",
}

WANDB_KWARGS = {
    "job_type": JOB_TYPE,
    "project": "neppo",
    "sync_tensorboard": True,
}


# --- Q-Critic ---


class QCritic(nn.Module):
    """Centralized Q-critic: Q(state, all_actions) -> scalar Q-value."""

    def __init__(self, state_dim, total_action_dim, hidden_dims, device="cpu"):
        super().__init__()
        layers = []
        input_dim = state_dim + total_action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
        self.to(device)

    def forward(self, state, all_actions):
        x = th.cat([state, all_actions], dim=-1)
        return self.net(x).squeeze(-1)


# --- MADDPG Runner ---


class MADDPGRunner:
    def __init__(self, agents, config, key, device=None):
        if device is None:
            device = "cuda" if th.cuda.is_available() else "cpu"

        self.device = device
        self.num_agents = len(agents)

        algo_kwargs = config["algo_kwargs"]
        actor_kwargs = config["actor_kwargs"]

        # Dimensions
        self.state_dim = sum(a.pi_obs_spec.shape[0] for a in agents)
        self.total_action_dim = sum(a.pi_action_spec.shape[0] for a in agents)

        # Create MLP actors
        keys = jax.random.split(key, len(agents) + 1)
        self.actors = []
        for agent, k in zip(agents, keys[:-1]):
            actor = MLPActor(
                agent.pi_obs_spec,
                agent.pi_action_spec,
                k,
                **{**actor_kwargs, "device": device},
            )
            self.actors.append(actor)

        # Create Q-critics (one per agent, centralized)
        critic_hidden = algo_kwargs.get("critic_hidden_dims", [64, 64])
        self.critics = [
            QCritic(self.state_dim, self.total_action_dim, critic_hidden, device)
            for _ in range(self.num_agents)
        ]

        # Target networks (PyTorch only - no JAX/Flax counterpart needed)
        self.target_actor_nets = [
            copy.deepcopy(actor.trl_actor_net) for actor in self.actors
        ]
        self.target_critics = [copy.deepcopy(critic) for critic in self.critics]
        for net in self.target_actor_nets:
            for p in net.parameters():
                p.requires_grad = False
        for tc in self.target_critics:
            for p in tc.parameters():
                p.requires_grad = False

        # Optimizers
        self.actor_optims = [
            th.optim.Adam(actor.trl_actor_net.parameters(), lr=algo_kwargs["lr_actor"])
            for actor in self.actors
        ]
        self.critic_optims = [
            th.optim.Adam(critic.parameters(), lr=algo_kwargs["lr_critic"])
            for critic in self.critics
        ]

        # Replay buffer
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=algo_kwargs["buffer_size"], device=device
            ),
            batch_size=algo_kwargs["batch_size"],
        )
        self.warmup_steps = algo_kwargs["warmup_steps"]
        self.total_stored = 0

        self.gamma = algo_kwargs["gamma"]
        self.tau = algo_kwargs["tau"]
        self.grad_norm_clip = algo_kwargs.get("grad_norm_clip", None)

    @property
    def policies(self):
        return self.actors

    def get_policy_contexts(self, key):
        return [
            actor.init_context(k)
            for actor, k in zip(self.actors, jax.random.split(key, len(self.actors)))
        ]

    def _store(self, tds):
        """Store per-agent TDs into replay buffer as a combined TD."""
        batch_size = tds[0].batch_size[0]

        source = {
            "state": tds[0]["state"],
            "next": {
                "state": tds[0]["next", "state"],
                "done": tds[0]["next", "done"],
            },
        }
        for i in range(self.num_agents):
            source[f"obs_{i}"] = tds[i]["observation"]
            source[f"action_{i}"] = tds[i]["action"]
            source[f"next_obs_{i}"] = tds[i]["next", "observation"]
            source[f"reward_{i}"] = tds[i]["next", "reward"]

        combined = TensorDict(source, batch_size=batch_size, device=self.device)
        self.buffer.extend(combined)
        self.total_stored += batch_size

    def _get_deterministic_action(self, actor, obs):
        """Get deterministic (mean) action from actor, differentiable."""
        loc = actor.trl_actor_net(obs)
        return th.tanh(loc)

    def _get_target_action(self, target_net, obs):
        """Get deterministic action from target actor net (no grad)."""
        loc = target_net(obs)
        return th.tanh(loc)

    def _soft_update(self):
        """Soft update target networks."""
        for actor, target_net in zip(self.actors, self.target_actor_nets):
            for p, tp in zip(actor.trl_actor_net.parameters(), target_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data * self.tau)
        for critic, target_critic in zip(self.critics, self.target_critics):
            for p, tp in zip(critic.parameters(), target_critic.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data * self.tau)

    def step(self, tds):
        """Training step. tds: list of per-agent TDs with 'state' injected."""
        self._store(tds)

        log_dict = {}

        # Log reward statistics for each agent from current transitions
        for i in range(self.num_agents):
            rewards = tds[i]["next", "reward"]
            dones = tds[i]["next", "done"]

            log_dict[f"collect/agent_{i}/step_rew_mean"] = rewards.mean().item()

            # Compute episode rewards (sum over each episode, then mean)
            num_episodes = dones.sum().item()
            if num_episodes > 0:
                log_dict[f"collect/agent_{i}/ep_rew_mean"] = (
                    rewards.sum().item() / num_episodes
                )

        if self.total_stored < self.warmup_steps:
            log_dict["debug/warmup"] = True
            log_dict["debug/buffer_size"] = self.total_stored
            return log_dict

        batch = self.buffer.sample()

        # Get current actions from buffer
        current_actions = [batch[f"action_{j}"] for j in range(self.num_agents)]
        current_all_actions = th.cat(current_actions, dim=-1)

        for i in range(self.num_agents):
            # --- Critic update ---
            with th.no_grad():
                target_actions = [
                    self._get_target_action(
                        self.target_actor_nets[j], batch[f"next_obs_{j}"]
                    )
                    for j in range(self.num_agents)
                ]
                target_all_actions = th.cat(target_actions, dim=-1)
                next_q = self.target_critics[i](
                    batch["next", "state"], target_all_actions
                )
                target_q = batch[f"reward_{i}"] + self.gamma * next_q * (
                    1 - batch["next", "done"].float()
                )

            current_q = self.critics[i](batch["state"], current_all_actions.detach())
            critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optims[i].zero_grad()
            critic_loss.backward()
            if self.grad_norm_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.critics[i].parameters(), self.grad_norm_clip
                )
            self.critic_optims[i].step()

            # --- Actor update ---
            # Replace agent i's action with current policy output (differentiable)
            new_actions = [a.detach() for a in current_actions]
            new_actions[i] = self._get_deterministic_action(
                self.actors[i], batch[f"obs_{i}"]
            )
            new_all_actions = th.cat(new_actions, dim=-1)
            actor_loss = -self.critics[i](
                batch["state"].detach(), new_all_actions
            ).mean()

            self.actor_optims[i].zero_grad()
            actor_loss.backward()
            if self.grad_norm_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.actors[i].trl_actor_net.parameters(), self.grad_norm_clip
                )
            self.actor_optims[i].step()

            log_dict[f"train/agent_{i}/critic_loss"] = critic_loss.item()
            log_dict[f"train/agent_{i}/actor_loss"] = actor_loss.item()
            log_dict[f"train/agent_{i}/q_value_mean"] = current_q.mean().item()

        # Soft update targets
        self._soft_update()

        log_dict["debug/buffer_size"] = self.total_stored

        return log_dict


# --- Training ---


def train(
    test=False,
    name=DEFAULT_NAME,
    job_type=JOB_TYPE,
    seed=SEED,
    config=CONFIG,
    wandb_kwargs=WANDB_KWARGS,
    device=None,
    gpu_num=0,
):
    if device is not None and not test:
        if device == "cuda":
            device = f"cuda:{gpu_num}"
            jax.config.update("jax_default_device", jax.devices("gpu")[gpu_num])
        config["device"] = device

    env_name = config["env"]["name"]
    exp_name = (
        f"{'test_' if test else ''}{env_name}_maddpg_{time.strftime('%m_%d-%H_%M_%S')}"
    )
    wandb_kwargs["job_type"] = job_type

    ENV_MAX_STEPS = config["env"]["max_steps"]

    if test:
        wandb_kwargs["offline"] = True
        wandb_kwargs["mode"] = "disabled"

        config["collect"]["num_envs"] = 2
        config["trainer"]["algo_kwargs"]["warmup_steps"] = 10
        config["trainer"]["algo_kwargs"]["buffer_size"] = 1000
        config["trainer"]["algo_kwargs"]["batch_size"] = 16
        config["device"] = "cpu"

        ENV_MAX_STEPS = 10

    if config["seed"] is not None:
        seed = config["seed"]
    else:
        seed = np.random.randint(2**32 - 1)

    key = jax.random.key(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    log_dir = f".logs/{exp_name}"
    save_dir = f".checkpoints/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)

    # Create JaxMARL environment
    env, agents, rewards, state_agent = make_jaxmarl_env(config["env"]["name"])

    sim = JaxMARLSim(env, agents, rewards, ENV_MAX_STEPS)

    # Create MADDPG runner
    key, runner_key = jax.random.split(key)
    runner = MADDPGRunner(
        agents, config["trainer"], runner_key, device=config["device"]
    )

    policy = PolicyAggregator(runner.policies)
    key, sk = jax.random.split(key)
    policy_context = policy.init_context(sk)

    collect = collect_factory(
        sim.reset,
        sim.step,
        policy,
        timesteps_per_env=TIMESTEPS_PER_ENV,
        num_envs=config["collect"]["num_envs"],
        extended_transition=True,
    )
    collect = jax.jit(collect)

    logger = ExtWandbLogger(
        exp_name=exp_name,
        **{
            **wandb_kwargs,
            "project": "neppo",
            "config": config,
        },
    )

    start_time = time.time()
    total_frames = 0
    next_save_frame = config["save"]["freq"]
    num_env_steps = config.get("num_env_steps", None)

    print("starting loop")
    while True:
        key, policy_key, collect_key = jax.random.split(key, 3)

        policy_context = policy.init_context(policy_key)

        # collect
        collect_start_time = time.time()
        transitions = collect(collect_key, policy_context)
        collect_end_time = time.time()

        tds = transitions_to_tensordicts(transitions)
        total_frames += transitions.step_count.size

        # Add global state for centralized critic
        state_obs, _ = jax.vmap(state_agent.obs_env2pi)(transitions.raw_obs)
        state_next_obs, _ = jax.vmap(state_agent.obs_env2pi)(transitions.raw_next_obs)
        state_obs_th = tdl(state_obs)
        state_next_obs_th = tdl(state_next_obs)
        for i in range(len(tds)):
            tds[i]["state"] = state_obs_th
            tds[i]["next", "state"] = state_next_obs_th

        # train
        train_start_time = time.time()
        log_dict = runner.step(tds)
        train_end_time = time.time()

        # logging
        log_dict.update(
            {
                "time/wall_min": (time.time() - start_time) / 60,
                "time/train_step_sec": train_end_time - train_start_time,
                "time/collect_step_sec": collect_end_time - collect_start_time,
                "time/sim_fps": transitions.step_count.size
                / (collect_end_time - collect_start_time),
                "time/total_frames": total_frames,
            }
        )

        logger.log_scalars(log_dict, timestep=total_frames)

        # save
        if total_frames >= next_save_frame:
            next_save_frame += config["save"]["freq"]
            for i, agent in enumerate(agents):
                save_actor(
                    runner.actors[i].trl_actor,
                    f"{save_dir}/actor_{total_frames}/{agent.name}.pth",
                )

        del transitions
        del tds

        if num_env_steps is not None and total_frames >= num_env_steps:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--num_env_steps", type=int, default=None)
    args = parser.parse_args()
    if args.env is not None:
        CONFIG["env"]["name"] = args.env
    if args.num_env_steps is not None:
        CONFIG["num_env_steps"] = args.num_env_steps
    train(args.test, device=args.device, gpu_num=args.gpu_num)
