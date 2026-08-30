"""Regret evaluation on the MPE environments (Sec. 4.3, Table 2, Fig. 3).

Regret of agent i for a saved joint policy: freeze every agent's actor (acting
deterministically), measure agent i's discounted return, then train a PPO best response for
agent i (warm-started from its own actor) against the other frozen agents and log

    regret/improvement = discounted return of the best response - frozen discounted return

after every PPO update. The number reported in the paper is the maximum of
``regret/improvement`` over best-response training, maximized over the agents
(``regret/max_improvement`` tracks the running maximum).

    for i in 0 1 2 3 4 5; do   # world comm: leadadversary_0, adversary_0..2, agent_0, agent_1
      python mpe_regret.py --env MPE_simple_world_comm_v3 --weights_dir pretrained/neppo_mixer_wc --agent_idx $i
    done

``--weights_dir`` holds one ``<agent_name>.pth`` per agent (``pretrained/*_wc`` or a
checkpoint written by the training scripts); the actor architecture is read from the file.
"""

import os
import re
import time

import jax
import numpy as np
import torch as th
from jaxtrl.rl.ppo._ppo import PPO
from jaxtrl.rl.ppo.actor.mlp import MLPActor
from jaxtrl.rl.ppo.critic import mlp_critic_factory
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import DeterministicWrapper, PolicyAggregator
from jaxtrl.utils.logger import ExtWandbLogger

from mpe_env import JaxMARLSim, make_jaxmarl_env
from marl.utils.collect import tdl, transitions_to_tensordicts
from marl.utils.torchrl import get_rollout_stats, save_actor

TIMESTEPS_PER_ENV = 25


def _extract_net_state_dict(saved_state_dict):
    """Extract trl_actor_net weights from a ProbabilisticActor state_dict."""
    net_sd = {}
    for key, val in saved_state_dict.items():
        m = re.search(r"(\d+)\.(weight|bias)$", key)
        if m:
            net_sd[f"{m.group(1)}.{m.group(2)}"] = val
            continue
        if key.endswith("log_std") and "log_std" not in net_sd:
            net_sd["log_std"] = val
    return net_sd


def _infer_net_arch(net_sd):
    """Hidden layer widths of an MLP actor from its state_dict (Linear layers at 0, 2, 4, ...)."""
    linear_idx = sorted(int(k.split(".")[0]) for k in net_sd if k.endswith(".weight"))
    return [net_sd[f"{i}.weight"].shape[0] for i in linear_idx[:-1]]


def load_actors(weights_dir, agents, jax_key, device="cpu"):
    """Reconstruct MLPActors (architecture read from the checkpoint) and load saved weights."""
    actors = []
    for i, agent in enumerate(agents):
        path = os.path.join(weights_dir, f"{agent.name}.pth")
        saved_sd = th.load(path, map_location=device, weights_only=True)
        net_sd = _extract_net_state_dict(saved_sd)
        actor = MLPActor(
            obs_space=agent.pi_obs_spec,
            action_space=agent.pi_action_spec,
            jax_key=jax.random.fold_in(jax_key, i),
            net_arch=_infer_net_arch(net_sd),
            activation_func="relu",
            distribution="tanh",
            distribution_kwargs={"tanh_loc": True},
            device=device,
        )
        actor.trl_actor_net.load_state_dict(net_sd)
        actors.append(actor)
    return actors


def discounted_return(td, gamma):
    """Mean discounted episode return of the transitions in td (sum over steps / episodes)."""
    discounted_reward = td["next", "reward"] * gamma ** td["next", "step_count"]
    return (discounted_reward.sum() / max(td["next", "done"].sum(), 1)).item()


def compute_baseline(collect_fn, key, train_idx, policy, baseline_samples, gamma):
    """Collect frames with all agents frozen, return the trainable agent's discounted return."""
    total_frames = 0
    returns = []

    key, sk = jax.random.split(key)
    policy_context = policy.init_context(sk)

    while total_frames < baseline_samples:
        key, sk = jax.random.split(key)
        transitions = collect_fn(sk, policy_context)
        total_frames += transitions.step_count.shape[0]

        tds = transitions_to_tensordicts(transitions)
        td = tds[train_idx]
        if td["next", "done"].any():
            returns.append(discounted_return(td, gamma))

        del transitions
        del tds

    baseline = float(np.mean(returns)) if returns else 0.0
    print(
        f"Baseline ({total_frames} frames, {len(returns)} collections): discounted_reward_sum = {baseline:.4f}"
    )
    return baseline


def train(
    weights_dir,
    agent_idx,
    env_name="MPE_simple_world_comm_v3",
    test=False,
    name=None,
    seed=None,
    device="cpu",
    gpu_num=0,
    num_envs=128,
    max_steps=25,
    num_env_steps=None,
    baseline_samples=100_000,
    save_freq=1_000_000,
    save_dir=None,
    lr=7e-4,
    num_epochs=10,
    use_state=True,
    random_init=False,
):
    if device == "cuda" and not test:
        device = f"cuda:{gpu_num}"
        jax.config.update("jax_default_device", jax.devices("gpu")[gpu_num])

    if test:
        num_envs = 2
        max_steps = 10
        device = "cpu"
        baseline_samples = 500

    if seed is None:
        seed = np.random.randint(2**32 - 1)

    key = jax.random.key(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    name_tag = f"_{name}" if name else ""
    init_tag = "_randominit" if random_init else ""
    exp_name = f"{'test_' if test else ''}{env_name}_regret_agent{agent_idx}{init_tag}{name_tag}_{time.strftime('%m_%d-%H_%M_%S')}"
    if save_dir is None:
        save_dir = f".checkpoints/{exp_name}"

    log_dir = f".logs/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)

    # --- Environment ---
    env, agents, rewards, state_agent = make_jaxmarl_env(env_name)
    assert (
        0 <= agent_idx < len(agents)
    ), f"agent_idx {agent_idx} out of range [0, {len(agents)})"

    sim = JaxMARLSim(env, agents, rewards, max_steps)

    # --- Load all actors; the frozen agents act deterministically ---
    key, actor_key = jax.random.split(key)
    actors = load_actors(weights_dir, agents, actor_key, device=device)
    for i in range(len(actors)):
        actors[i] = DeterministicWrapper(actors[i])

    # --- Create PPO for trainable agent only ---
    train_agent = agents[agent_idx]
    critic_kwargs = {
        "net_arch": [64, 64, 64],
        "activation_func": "relu",
        "device": device,
    }
    if use_state:
        critic_kwargs["input_key"] = "state"
        critic_kwargs["input_size_override"] = state_agent._state_dim
    critic = mlp_critic_factory(
        train_agent.pi_obs_spec,
        train_agent.pi_action_spec,
        **critic_kwargs,
    )

    rl_kwargs = {
        "sub_batch_size": TIMESTEPS_PER_ENV * num_envs,
        "num_epochs": num_epochs,
        "lr": lr,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "critic_coef": 0.5,
        "entropy_eps": 0.005,
        "loss_critic_type": "smooth_l1",
        "max_grad_norm": 10.0,
        "clip_grad_norm": True,
    }

    # --- Baseline: every agent frozen and deterministic ---
    policy = PolicyAggregator(actors)
    collect = collect_factory(
        sim.reset,
        sim.step,
        policy,
        timesteps_per_env=TIMESTEPS_PER_ENV,
        num_envs=num_envs,
        extended_transition=True,
    )
    collect = jax.jit(collect)

    print(f"Computing baseline for agent {agent_idx} ({train_agent.name})...")
    key, baseline_key = jax.random.split(key)
    baseline_reward = compute_baseline(
        collect, baseline_key, agent_idx, policy, baseline_samples, rl_kwargs["gamma"]
    )

    # --- Best response: the trained agent samples stochastically, the others stay frozen ---
    if random_init:
        key, reinit_key = jax.random.split(key)
        actors[agent_idx] = MLPActor(
            obs_space=train_agent.pi_obs_spec,
            action_space=train_agent.pi_action_spec,
            jax_key=reinit_key,
            net_arch=[64],
            activation_func="relu",
            distribution="tanh",
            distribution_kwargs={"tanh_loc": True},
            device=device,
        )
    else:
        # warm start from the saved actor with a unit exploration std
        actors[agent_idx] = actors[agent_idx].policy
        for pname, param in actors[agent_idx].trl_actor_net.named_parameters():
            if pname == "log_std":
                param.data.zero_()

    policy = PolicyAggregator(actors)
    collect = collect_factory(
        sim.reset,
        sim.step,
        policy,
        timesteps_per_env=TIMESTEPS_PER_ENV,
        num_envs=num_envs,
        extended_transition=True,
    )
    collect = jax.jit(collect)

    ppo = PPO(actors[agent_idx], critic, device=device, **rl_kwargs)

    # --- Logger ---
    wandb_kwargs = {
        "job_type": "regret",
        "project": "neppo",
        "config": {
            "env": env_name,
            "agent_idx": agent_idx,
            "agent_name": train_agent.name,
            "weights_dir": weights_dir,
            "baseline_reward": baseline_reward,
            "rl_kwargs": rl_kwargs,
            "num_envs": num_envs,
            "max_steps": max_steps,
            "seed": seed,
            "use_state": use_state,
            "random_init": random_init,
        },
    }
    if test:
        wandb_kwargs["offline"] = True
        wandb_kwargs["mode"] = "disabled"

    logger = ExtWandbLogger(exp_name=exp_name, **wandb_kwargs)

    # --- Training loop ---
    start_time = time.time()
    total_frames = 0
    next_save_frame = save_freq
    max_improvement = -float("inf")

    init_type = "random" if random_init else "warm-started"
    print(
        f"Training agent {agent_idx} ({train_agent.name}), {init_type}, baseline discounted_reward_sum = {baseline_reward:.4f}"
    )
    print("starting loop")
    while True:
        # Sync trainable actor weights to JAX
        key, policy_key, collect_key = jax.random.split(key, 3)
        policy_context = policy.init_context(policy_key)

        # Collect
        collect_start_time = time.time()
        transitions = collect(collect_key, policy_context)
        collect_end_time = time.time()

        tds = transitions_to_tensordicts(transitions)
        total_frames += transitions.step_count.shape[0]

        td = tds[agent_idx]

        # Add centralized state for critic if enabled
        if use_state:
            state_obs, _ = jax.vmap(state_agent.obs_env2pi)(transitions.raw_obs)
            state_next_obs, _ = jax.vmap(state_agent.obs_env2pi)(transitions.raw_next_obs)
            td["state"] = tdl(state_obs)
            td["next", "state"] = tdl(state_next_obs)

        # Train
        train_start_time = time.time()
        log_dict = ppo.step(td)
        train_end_time = time.time()

        # Logging: regret = improvement of the best response's discounted return
        log_dict["baseline/discounted_reward_sum"] = baseline_reward
        br_return = discounted_return(td, rl_kwargs["gamma"])
        max_improvement = max(max_improvement, br_return - baseline_reward)
        log_dict["regret/discounted_reward_sum"] = br_return
        log_dict["regret/improvement"] = br_return - baseline_reward
        log_dict["regret/max_improvement"] = max_improvement
        log_dict.update(
            {
                "time/wall_min": (time.time() - start_time) / 60,
                "time/train_step_sec": train_end_time - train_start_time,
                "time/collect_step_sec": collect_end_time - collect_start_time,
                "time/sim_fps": transitions.step_count.shape[0]
                / (collect_end_time - collect_start_time),
                "time/total_frames": total_frames,
            }
        )

        logger.log_scalars(log_dict, timestep=total_frames)

        # Save
        if total_frames >= next_save_frame:
            next_save_frame += save_freq
            save_actor(
                actors[agent_idx].trl_actor,
                f"{save_dir}/actor_{total_frames}/{train_agent.name}.pth",
            )

        del transitions
        del tds

        if num_env_steps is not None and total_frames >= num_env_steps:
            break

    print(
        f"agent {agent_idx} ({train_agent.name}): max regret/improvement over training = {max_improvement:.4f} "
        f"({total_frames} frames)"
    )
    return max_improvement


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train best response for one agent (regret computation)"
    )
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--agent_idx", type=int, required=True)
    parser.add_argument("--env", type=str, default="MPE_simple_world_comm_v3")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--num_env_steps", type=int, default=None)
    parser.add_argument("--baseline_samples", type=int, default=100_000)
    parser.add_argument("--save_freq", type=int, default=1_000_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--use_state", action="store_true", default=True)
    parser.add_argument("--no_state", dest="use_state", action="store_false")
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="start the best response from random weights instead of the saved actor",
    )
    args = parser.parse_args()

    train(
        weights_dir=args.weights_dir,
        agent_idx=args.agent_idx,
        env_name=args.env,
        test=args.test,
        name=args.name,
        seed=args.seed,
        device=args.device,
        gpu_num=args.gpu_num,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        num_env_steps=args.num_env_steps,
        baseline_samples=args.baseline_samples,
        save_freq=args.save_freq,
        save_dir=args.save_dir,
        lr=args.lr,
        num_epochs=args.num_epochs,
        use_state=args.use_state,
        random_init=args.random_init,
    )
