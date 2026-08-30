"""IPPO training on a 2-player matrix game."""

import os
import time

import jax
import numpy as np
import torch as th
from jaxtrl.marl.ippo import IPPOTrainer
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator
from jaxtrl.utils.logger import ExtWandbLogger
from jaxtrl.utils.tensordict import transitions_to_ma_tensordict

from mpe_env import JaxMARLSim
from marl.utils.torchrl import save_actor
from matrix_game_env import make_matrix_game_env

DEFAULT_NAME = "matrix_ippo"
JOB_TYPE = "matrix_ippo"
SEED = None

TIMESTEPS_PER_ENV = 1
NUM_ENVS = 10000

CONFIG = {
    "trainer": {
        "actor": "mlp",
        "critic": "mlp",
        "actor_kwargs": {
            "net_arch": [32, 32],
            "activation_func": "relu",
            "distribution": "tanh",
            "distribution_kwargs": {"tanh_loc": True},
        },
        "critic_kwargs": {
            "net_arch": [32, 32],
            "activation_func": "relu",
        },
        "rl_kwargs": {
            "entropy_eps": 0.0,
            "num_epochs": 20,
            "sub_batch_size": TIMESTEPS_PER_ENV * NUM_ENVS,
            "gamma": 0.0,
            "gae_lambda": 0.0,
            "clip_epsilon": 0.2,
            "critic_coef": 0.5,
            "lr": 1e-3,
            "loss_critic_type": "smooth_l1",
            "max_grad_norm": 10.0,
            "clip_grad_norm": True,
        },
    },
    "collect": {
        "num_envs": NUM_ENVS,
    },
    "save": {
        "freq": 100_000,
    },
    "env": {
        "max_steps": 1,
    },
    "seed": None,
    "device": "cpu",
}

WANDB_KWARGS = {
    "job_type": JOB_TYPE,
    "project": "neppo",
    "sync_tensorboard": True,
}


def train(
    test=False,
    name=DEFAULT_NAME,
    job_type=JOB_TYPE,
    seed=SEED,
    config=CONFIG,
    wandb_kwargs=WANDB_KWARGS,
    device=None,
    gpu_num=0,
    alpha=0.0,
    lr_decay_inverse_sqrt=False,
):
    if device is not None and not test:
        if device == "cuda":
            device = f"cuda:{gpu_num}"
            jax.config.update("jax_default_device", jax.devices("gpu")[gpu_num])
        else:
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
        config["device"] = device

    name_tag = f"_{name}" if name else ""
    alpha_tag = f"_alpha_{alpha}"
    exp_name = f"{'test_' if test else ''}matrix_game_ippo{alpha_tag}{name_tag}_{time.strftime('%m_%d-%H_%M_%S')}"
    wandb_kwargs["job_type"] = job_type

    ENV_MAX_STEPS = config["env"]["max_steps"]

    if test:
        wandb_kwargs["offline"] = True
        wandb_kwargs["mode"] = "disabled"
        config["collect"]["num_envs"] = 2
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

    # Create matrix game environment
    env, agents, rewards, _state_agent = make_matrix_game_env(alpha=alpha)

    sim = JaxMARLSim(env, agents, rewards, ENV_MAX_STEPS)

    # Create IPPO trainer (decentralized critic, individual rewards)
    key, trainer_key = jax.random.split(key)
    trainer = IPPOTrainer(
        agents, config["trainer"], trainer_key, device=config["device"]
    )

    policy = PolicyAggregator(trainer.policies)
    key, sk = jax.random.split(key)
    policy_context = policy.init_context(sk)

    collect = collect_factory(
        sim.reset,
        sim.step,
        policy,
        timesteps_per_env=TIMESTEPS_PER_ENV if not test else ENV_MAX_STEPS,
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

    # Learning rate decay tracking
    base_lr = config["trainer"]["rl_kwargs"]["lr"]
    train_step_count = 0

    print("starting matrix game IPPO loop")
    while True:
        key, policy_key, collect_key = jax.random.split(key, 3)

        policy_context = policy.init_context(policy_key)

        # collect
        collect_start_time = time.time()
        transitions = collect(collect_key, policy_context)
        collect_end_time = time.time()

        ma_td = transitions_to_ma_tensordict(transitions)
        total_frames += transitions.step_count.size

        # train (no centralized critic, no shared reward)
        train_start_time = time.time()

        # Apply inverse square root learning rate decay
        if lr_decay_inverse_sqrt:
            train_step_count += 1
            new_lr = base_lr / (train_step_count ** 0.5)
            for ppo in trainer.algo.ppos:
                for param_group in ppo.optim.param_groups:
                    param_group["lr"] = new_lr

        log_dict = trainer.step(ma_td)
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
        if lr_decay_inverse_sqrt:
            log_dict["train/lr"] = new_lr

        # Override state with player strategies for quadratic phi
        # actions are in tanh space [-1,1], strategies are in [0,1]
        strategies = th.cat(
            [(ma_td[f"agent_{j}"]["action"] + 1.0) / 2.0 for j in range(len(agents))], dim=-1
        )  # (batch, 2)
        log_dict[f"strategies/player_A/mean"] = strategies[:, 0].mean().item()
        log_dict[f"strategies/player_A/std"] = strategies[:, 0].std().item()
        log_dict[f"strategies/player_B/mean"] = strategies[:, 1].mean().item()
        log_dict[f"strategies/player_B/std"] = strategies[:, 1].std().item()

        logger.log_scalars(log_dict, timestep=total_frames)

        # save
        if total_frames >= next_save_frame:
            next_save_frame += config["save"]["freq"]
            for i, agent in enumerate(agents):
                save_actor(
                    trainer.actors[i].trl_actor,
                    f"{save_dir}/actor_{total_frames}/{agent.name}.pth",
                )

        del transitions
        del ma_td

        if num_env_steps is not None and total_frames >= num_env_steps:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_env_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.0, help="game parameter in [0, 1]")
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--lr_decay_inverse_sqrt", action="store_true")
    args = parser.parse_args()

    if args.num_envs is not None:
        CONFIG["collect"]["num_envs"] = args.num_envs
        CONFIG["trainer"]["rl_kwargs"]["sub_batch_size"] = (
            TIMESTEPS_PER_ENV * args.num_envs
        )
    if args.lr is not None:
        CONFIG["trainer"]["rl_kwargs"]["lr"] = args.lr
    if args.num_env_steps is not None:
        CONFIG["num_env_steps"] = args.num_env_steps
    if args.save_freq is not None:
        CONFIG["save"]["freq"] = args.save_freq

    train(
        args.test,
        name=args.name,
        device=args.device,
        gpu_num=args.gpu_num,
        alpha=args.alpha,
        lr_decay_inverse_sqrt=args.lr_decay_inverse_sqrt,
    )
