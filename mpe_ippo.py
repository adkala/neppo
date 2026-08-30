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

from mpe_env import JaxMARLSim, make_jaxmarl_env
from marl.utils.torchrl import save_actor

DEFAULT_NAME = "jaxmarl_ippo"
JOB_TYPE = "jaxmarl_ippo"
SEED = None

TIMESTEPS_PER_ENV = 25
NUM_ENVS = 100

CONFIG = {
    "trainer": {
        "actor": "mlp",
        "critic": "mlp",
        "actor_kwargs": {
            "net_arch": [64, 64],
            "activation_func": "relu",
            "distribution": "tanh",
            "distribution_kwargs": {"tanh_loc": True},
        },
        "critic_kwargs": {
            "net_arch": [64, 64],
            "activation_func": "relu",
        },
        "rl_kwargs": {
            "entropy_eps": 0.01,
            "num_epochs": 10,
            "sub_batch_size": TIMESTEPS_PER_ENV * NUM_ENVS,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "critic_coef": 1.0,
            "lr": 7e-4,
            "loss_critic_type": "smooth_l1",
            "max_grad_norm": 10.0,
            "clip_grad_norm": True,
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
        f"{'test_' if test else ''}{env_name}_ippo_{time.strftime('%m_%d-%H_%M_%S')}"
    )
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

    # Create JaxMARL environment
    env, agents, rewards, _state_agent = make_jaxmarl_env(
        config["env"]["name"], local_ratio=0.0
    )

    sim = JaxMARLSim(env, agents, rewards, ENV_MAX_STEPS)

    # Create IPPO trainer
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

        ma_td = transitions_to_ma_tensordict(transitions)
        total_frames += transitions.step_count.size

        # train
        train_start_time = time.time()
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_env_steps", type=int, default=None)
    args = parser.parse_args()
    if args.env is not None:
        CONFIG["env"]["name"] = args.env
    if args.num_epochs is not None:
        CONFIG["trainer"]["rl_kwargs"]["num_epochs"] = args.num_epochs
    if args.num_env_steps is not None:
        CONFIG["num_env_steps"] = args.num_env_steps
    train(args.test, device=args.device, gpu_num=args.gpu_num)
