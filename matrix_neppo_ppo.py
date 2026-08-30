"""NePPO with HAPPO / PPO inner solvers on the two-player matrix games of Section 4.

This is the sampled pipeline of Algorithm 1 (marl/algorithms/neppo/, the implementation used
on Simple World Comm) applied to the 2x2 games of eq. (7): CoopGameSolver is HAPPO on the
shared reward Phi_w, RLSolver is PPO on each player's own reward, F_i is estimated from
rollouts, and w takes a two-point zeroth-order step.

  --phi convex      Phi_w = w J_1 + (1 - w) J_2              (Sec. 4.1, Fig. 1)
  --phi quadratic   Phi_w(x, y) = -(x - p)^2 - (y - q)^2     (the potential of Sec. 4.2)

where x and y are the probabilities of player 1 choosing A2 and player 2 choosing B2.
Table 1 / Fig. 2 were produced with the closed-form inner solvers of matrix_neppo.py; this
script is the sampled counterpart. CONFIG holds the Sec. 4.1 (Fig. 1) hyperparameters;
QUADRATIC_CONFIG the ones the sampled pipeline needs to recover Table 1's equilibria with
the quadratic potential (applied by default for --phi quadratic, see the README). Every
value can be overridden from the command line.
"""

import copy
import json
import os
import time

import jax
import numpy as np
import torch as th
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator
from jaxtrl.utils.logger import ExtWandbLogger
from torch import nn

from mpe_env import JaxMARLSim
from marl.algorithms.neppo.runner import NePPORunner
from marl.utils.collect import transitions_to_tensordicts
from marl.utils.torchrl import get_rollout_stats, save_actor
from matrix_game_env import make_matrix_game_env


class QuadraticPhi(nn.Module):
    """Phi_w(x, y) = -(x - p)^2 - (y - q)^2 with learnable w = (p, q)."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(th.tensor(0.5))
        self.q = nn.Parameter(th.tensor(0.5))

    def project(self):
        with th.no_grad():
            self.p.clamp_(0.0, 1.0)
            self.q.clamp_(0.0, 1.0)

    def forward(self, state, reward):
        # state: (batch, 2) = the two players' strategies in [0, 1]
        x = state[:, 0]
        y = state[:, 1]
        return -((x - self.p) ** 2) - (y - self.q) ** 2


JOB_TYPE = "matrix_neppo_ppo"

TIMESTEPS_PER_ENV = 1
NUM_ENVS = 10000
NUM_PHI_UPDATES = 500


def default_num_env_steps(config):
    """Env steps for NUM_PHI_UPDATES potential updates: one update takes
    2 * pm_zo_steps + 4 * br_zo_steps + 6 collects of NUM_ENVS steps (two players)."""
    ak = config["trainer"]["algo_kwargs"]
    collects = 2 * ak["pm_zo_steps"] + 4 * ak["br_zo_steps"] + 6
    return NUM_PHI_UPDATES * collects * config["collect"]["num_envs"] * TIMESTEPS_PER_ENV


CONFIG = {
    "trainer": {
        "algo": "neppo",
        "actor": "mlp",
        "critic": "mlp",
        "algo_kwargs": {
            "pm_zo_steps": 1,
            "br_zo_steps": 1,
            #
            "phi_lr": 1e-2,
            "phi_epochs": 1,
            "zeroth_order_delta": 1e-3,
            "phi_max_grad_norm": 10.0,
            "phi_clip_grad_norm": True,
            #
            "entropy_eps": 0.0,
            "num_epochs": 10,
            "gamma": 0.95,
            "gae_lambda": 0.99,
            "clip_epsilon": 0.2,
            "critic_coef": 1,
            "lr": 1e-3,
            "loss_critic_type": "smooth_l1",
            # mixer=False -> NPLambda, the convex-combination potential of Sec. 4.1
            "mixer": False,
            "mixer_input_dim": 1,
            "mixer_output_dim": 2,
        },
        "actor_kwargs": {
            "net_arch": [1],
            "activation_func": "relu",
            "distribution": "tanh",
            "distribution_kwargs": {"tanh_loc": True},
        },
        "critic_kwargs": {
            "net_arch": [1],
            "activation_func": "relu",
            "input_key": "state",
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

# Hyperparameters for the quadratic potential of Sec. 4.2 (--phi quadratic), found by a sweep
# over the alpha games (see README, "Table 1 with the sampled solvers"). What each one does:
#   pm_zo_steps / br_zo_steps = 5   K1 / K2 of Algorithm 1: both warm-started solves have to
#                                   converge for F_hat - F_check to be a finite difference
#   entropy_eps = 0.01              keeps the two HAPPO solves stochastic and mutually consistent
#                                   (without it F_hat - F_check is dominated by solver noise)
#   zeroth_order_delta = 0.01       just above the solver's resolution; the returned policy is the
#                                   solution under w -/+ delta u, so it sits ~delta from (p, q) and
#                                   its regret at alpha = 1 (matching pennies) is ~1.8 delta
#   phi_lr = 0.1, clip 2, 1/sqrt t  outer step: at most 0.2 initially, decaying as 1/sqrt(1 + t)
#                                   in potential updates (phi_lr_inverse_sqrt); larger constant
#                                   steps make (p, q) bounce because the check solve is warm-started
#                                   from the previous iteration
#   grpo_pm = False, no advantage normalization, no actor/critic gradient clipping, lr 0.01,
#   20 epochs, critic_coef 0.5, [32, 32] networks: the plain HAPPO / PPO inner solvers
#   critics on the observation (--critic_input observation): a critic that sees the joint
#                                   strategies fits the reward exactly and the best response stops
#                                   learning; projection of (p, q) onto [0, 1]^2 (--project)
QUADRATIC_CONFIG = {
    "algo_kwargs": {
        "pm_zo_steps": 5,
        "br_zo_steps": 5,
        "phi_lr": 0.1,
        "phi_max_grad_norm": 2.0,
        "phi_lr_inverse_sqrt": True,
        "zeroth_order_delta": 0.01,
        "entropy_eps": 0.01,
        "num_epochs": 20,
        "critic_coef": 0.5,
        "lr": 0.01,
        "grpo_pm": False,
        "normalize_advantage": False,
        "actor_clip_grad_norm": False,
        "critic_clip_grad_norm": False,
    },
    "actor_kwargs": {"net_arch": [32, 32]},
    "critic_kwargs": {"net_arch": [32, 32]},
}
QUADRATIC_CRITIC_INPUT = "observation"
QUADRATIC_PROJECT = True


def apply_quadratic_config(config):
    """Merge QUADRATIC_CONFIG into config['trainer'] (in place)."""
    for section, values in QUADRATIC_CONFIG.items():
        config["trainer"][section].update(values)
    return config


WANDB_KWARGS = {
    "job_type": JOB_TYPE,
    "project": "neppo",
    "sync_tensorboard": True,
}


def train(
    test=False,
    name=None,
    job_type=JOB_TYPE,
    config=CONFIG,
    wandb_kwargs=WANDB_KWARGS,
    device=None,
    gpu_num=0,
    alpha=0.0,
    phi_type="quadratic",
    phi_scheduler=False,
    phi_scheduler_unit="update",
    critic_input="state",
    project=False,
):
    if device is not None and not test:
        if device == "cuda":
            device = f"cuda:{gpu_num}"
            jax.config.update("jax_default_device", jax.devices("gpu")[gpu_num])
        else:
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
        config["device"] = device

    name_tag = f"_{name}" if name else ""
    exp_name = f"{'test_' if test else ''}matrix_game_neppo_ppo_{phi_type}_alpha_{alpha}{name_tag}_{time.strftime('%m_%d-%H_%M_%S')}"
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

    env, agents, rewards, state_agent = make_matrix_game_env(alpha=alpha)

    # The potential sees the "state" = the pair of player strategies; the critics see either
    # the state (default) or the constant observation (--critic_input observation).
    config["trainer"]["critic_kwargs"]["input_key"] = critic_input
    config["trainer"]["critic_kwargs"]["input_size_override"] = (
        state_agent._state_dim if critic_input == "state" else None
    )
    config["trainer"]["algo_kwargs"]["mixer_input_dim"] = state_agent._state_dim
    config["trainer"]["algo_kwargs"]["mixer_output_dim"] = len(agents)

    sim = JaxMARLSim(env, agents, rewards, ENV_MAX_STEPS)

    runner = NePPORunner(agents, config["trainer"], key, device=config["device"])

    if phi_type == "quadratic":
        # Replace the default (convex-combination) potential with Phi_w(x,y) = -(x-p)^2 - (y-q)^2
        phi = QuadraticPhi().to(config["device"])
        runner.algo.phi = phi
        runner.algo.phi_optim = th.optim.SGD(
            phi.parameters(), lr=config["trainer"]["algo_kwargs"]["phi_lr"]
        )
        runner.algo.phi_num_params = sum(p.numel() for p in phi.parameters())
        # Update the copies held by the PM / BR steps
        runner.algo.pm.phi = copy.deepcopy(phi)
        runner.algo.br.phi = copy.deepcopy(phi)
    # Potential step-size decay 1 / sqrt(1 + t). NePPO.phi_update sets the optimizer's lr from
    # hp.phi_lr before every step (a torch LR scheduler on phi_optim would be overwritten), so
    # the decay is driven through hp: per potential update via the built-in
    # phi_lr_inverse_sqrt, per collect by rescaling hp.phi_lr in the loop below.
    phi_lr0 = config["trainer"]["algo_kwargs"]["phi_lr"]
    if phi_scheduler and phi_scheduler_unit == "update":
        runner.algo.hp.phi_lr_inverse_sqrt = True
    num_collects = 0

    policy = PolicyAggregator(runner.policies)
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

    print("starting matrix game NePPO loop")
    while True:
        key, sk = jax.random.split(key)
        policy_contexts = runner.get_policy_contexts(sk)
        policy_context[:] = policy_contexts

        # collect
        key, sk = jax.random.split(key)
        collect_start_time = time.time()
        transitions = collect(sk, policy_context)
        collect_end_time = time.time()

        tds = transitions_to_tensordicts(transitions)
        total_frames += transitions.step_count.shape[0]

        # The state is the pair of player strategies (x, y) in [0, 1]^2.
        # Actions are in tanh space [-1, 1].
        strategies = th.cat(
            [(tds[j]["action"] + 1.0) / 2.0 for j in range(len(tds))], dim=-1
        )  # (batch, 2)
        for i in range(len(tds)):
            tds[i]["state"] = strategies
            tds[i]["next", "state"] = strategies

        rewards_cache = {}
        for i in range(len(tds)):
            rewards_cache[i] = tds[i]["next", "reward"]

        # train
        train_start_time = time.time()
        log_dict = runner.step(tds)
        if project and phi_type == "quadratic":
            runner.algo.phi.project()
        num_collects += 1
        if phi_scheduler and phi_scheduler_unit == "collect":
            runner.algo.hp.phi_lr = phi_lr0 / (1 + num_collects) ** 0.5
        train_end_time = time.time()

        # logging
        branch = log_dict["debug/branch"]
        pre = {0: "pm", 1: "hat_pm", 2: "br", 3: "hat_br", 4: "pd", 5: "hat_pd"}.get(
            branch, ""
        )
        if branch in {2, 3, 4, 5}:
            agent_idx = log_dict.get(f"{pre}/rollout/agent_idx")
            pre = f"{pre}/t{agent_idx}"

        if phi_type == "quadratic":
            log_dict["phi/p"] = runner.algo.phi.p.item()
            log_dict["phi/q"] = runner.algo.phi.q.item()
            log_dict["phi/lr"] = runner.algo._get_phi_lr()
        else:
            # weight w on player 1's reward in Phi_w = w J_1 + (1 - w) J_2
            log_dict["phi/w"] = th.softmax(runner.algo.phi.logits, dim=0)[0].item()

        log_dict[f"strategies/{pre}/player_A/mean"] = strategies[:, 0].mean().item()
        log_dict[f"strategies/{pre}/player_A/std"] = strategies[:, 0].std().item()
        log_dict[f"strategies/{pre}/player_B/mean"] = strategies[:, 1].mean().item()
        log_dict[f"strategies/{pre}/player_B/std"] = strategies[:, 1].std().item()

        for j in range(len(tds)):
            tds[j]["next", "reward"] = rewards_cache[j]
            log_dict |= {
                f"{pre}/rollout/{j}/{k}": v
                for k, v in get_rollout_stats(tds[j], include_len=j == 0).items()
            }

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

        # save
        if total_frames >= next_save_frame:
            next_save_frame += config["save"]["freq"]
            for i, agent in enumerate(agents):
                save_actor(
                    runner.algo.pm.actors[i].trl_actor,
                    f"{save_dir}/actor_{total_frames}/{agent.name}.pth",
                )
            save_actor(
                runner.algo.phi,
                f"{save_dir}/actor_{total_frames}/phi.pth",
            )

        del transitions
        del tds

        if num_env_steps is not None and total_frames >= num_env_steps:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--phi",
        type=str,
        default="quadratic",
        choices=["convex", "quadratic"],
        help="potential parameterization: convex (Sec. 4.1) or quadratic (Sec. 4.2)",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="game parameter in [0, 1]")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=None,
        help=f"default: {NUM_PHI_UPDATES} potential updates",
    )
    parser.add_argument("--lr", type=float, default=None, help="actor / critic learning rate")
    parser.add_argument("--phi_lr", type=float, default=None, help="potential step size eta")
    parser.add_argument(
        "--phi_max_grad_norm", type=float, default=None, help="clip the zeroth-order gradient"
    )
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument(
        "--phi_scheduler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="decay the potential step size as 1/sqrt(1 + t) (default: on for --phi quadratic)",
    )
    parser.add_argument(
        "--phi_scheduler_unit",
        type=str,
        default="update",
        choices=["update", "collect"],
        help="count t of the 1/sqrt(1 + t) decay in potential updates (default) or in collects",
    )
    parser.add_argument("--pm_zo_steps", type=int, default=None, help="K1 of Algorithm 1")
    parser.add_argument("--br_zo_steps", type=int, default=None, help="K2 of Algorithm 1")
    parser.add_argument("--zeroth_order_delta", type=float, default=None, help="delta of Algorithm 1")
    parser.add_argument("--entropy_eps", type=float, default=None, help="PPO entropy bonus")
    parser.add_argument("--num_epochs", type=int, default=None, help="PPO epochs per step")
    parser.add_argument(
        "--critic_input",
        type=str,
        default=None,
        choices=["state", "observation"],
        help="critic input: the joint strategies (default for --phi convex) or the constant "
        "observation (default for --phi quadratic)",
    )
    parser.add_argument(
        "--project",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="quadratic potential: clamp w = (p, q) to [0, 1]^2 after every step (default: on)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="override any CONFIG['trainer'] entry, e.g. --set algo_kwargs.gamma=0.99",
    )
    args = parser.parse_args()

    if args.phi == "quadratic":
        apply_quadratic_config(CONFIG)
    if args.critic_input is None:
        args.critic_input = QUADRATIC_CRITIC_INPUT if args.phi == "quadratic" else "state"
    if args.project is None:
        args.project = QUADRATIC_PROJECT if args.phi == "quadratic" else False
    if args.phi_scheduler is None:
        args.phi_scheduler = CONFIG["trainer"]["algo_kwargs"].get("phi_lr_inverse_sqrt", False)
    # the decay is applied through phi_scheduler / phi_scheduler_unit in train()
    CONFIG["trainer"]["algo_kwargs"]["phi_lr_inverse_sqrt"] = False

    if args.seed is not None:
        CONFIG["seed"] = args.seed
    if args.num_envs is not None:
        CONFIG["collect"]["num_envs"] = args.num_envs
    if args.lr is not None:
        CONFIG["trainer"]["algo_kwargs"]["lr"] = args.lr
    if args.phi_lr is not None:
        CONFIG["trainer"]["algo_kwargs"]["phi_lr"] = args.phi_lr
    if args.phi_max_grad_norm is not None:
        CONFIG["trainer"]["algo_kwargs"]["phi_max_grad_norm"] = args.phi_max_grad_norm
    if args.save_freq is not None:
        CONFIG["save"]["freq"] = args.save_freq
    if args.pm_zo_steps is not None:
        CONFIG["trainer"]["algo_kwargs"]["pm_zo_steps"] = args.pm_zo_steps
    if args.br_zo_steps is not None:
        CONFIG["trainer"]["algo_kwargs"]["br_zo_steps"] = args.br_zo_steps
    if args.zeroth_order_delta is not None:
        CONFIG["trainer"]["algo_kwargs"]["zeroth_order_delta"] = args.zeroth_order_delta
    if args.entropy_eps is not None:
        CONFIG["trainer"]["algo_kwargs"]["entropy_eps"] = args.entropy_eps
    if args.num_epochs is not None:
        CONFIG["trainer"]["algo_kwargs"]["num_epochs"] = args.num_epochs
    for kv in args.set:
        key, value = kv.split("=", 1)
        node = CONFIG["trainer"]
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = json.loads(value)
    if args.num_env_steps is not None:
        CONFIG["num_env_steps"] = args.num_env_steps
    else:
        CONFIG["num_env_steps"] = default_num_env_steps(CONFIG)

    print(
        f"phi={args.phi} alpha={args.alpha} critic_input={args.critic_input} project={args.project} "
        f"phi_scheduler={args.phi_scheduler} ({args.phi_scheduler_unit}) "
        f"num_env_steps={CONFIG['num_env_steps']}"
    )
    print("trainer config:", json.dumps(CONFIG["trainer"]))

    train(
        args.test,
        name=args.name,
        device=args.device,
        gpu_num=args.gpu_num,
        alpha=args.alpha,
        phi_type=args.phi,
        phi_scheduler=args.phi_scheduler,
        phi_scheduler_unit=args.phi_scheduler_unit,
        critic_input=args.critic_input,
        project=args.project,
    )
