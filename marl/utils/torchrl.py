import os

import gymnasium as gym
import torch as th
from torchrl.data.tensor_specs import Bounded, Categorical, MultiCategorical


def gym_to_torchrl_spec(space: gym.Space, device="cpu"):
    if isinstance(space, gym.spaces.Box):
        return Bounded(
            low=th.tensor(space.low, dtype=th.float32),
            high=th.tensor(space.high, dtype=th.float32),
            shape=space.shape,
            device=device,
            dtype=th.float32,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return Categorical(n=space.n, shape=(), dtype=th.long, device=device)
    elif isinstance(space, gym.spaces.MultiBinary):
        return Categorical(n=2, shape=(space.n,), dtype=th.long, device=device)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return MultiCategorical(
            nvec=th.tensor(space.nvec),
            shape=space.nvec.shape,
            dtype=th.long,
            device=device,
        )
    else:
        raise NotImplementedError(f"Space type {type(space)} not supported")


def save_actor(actor, path):
    # Create directory if it does not exist
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # Handle lists of optimizers/actors
    if isinstance(actor, list):
        state_dicts = [item.state_dict() for item in actor]
        th.save(state_dicts, path)
    else:
        # Save the actor using torch.save
        th.save(actor.state_dict(), path)


def get_rollout_stats(tensordict_data, include_len=True):
    done = tensordict_data.get(("next", "done"))

    rollout_data = (
        {
            "ep_rew_mean": tensordict_data["next", "episode_reward"][done]
            .mean()
            .item(),
            "step_rew_mean": tensordict_data["next", "reward"].mean().item(),
        }
        if done.any()
        else {}
    )

    if include_len:
        rollout_data["ep_len_mean"] = (
            tensordict_data["next", "step_count"][done].float().mean().item()
        )
        rollout_data["ep_rew_mean"] = (
            tensordict_data["next", "episode_reward"][done].mean().item()
        )
        rollout_data["step_rew_mean"] = tensordict_data["next", "reward"].mean().item()

    return rollout_data


def get_explained_variance(state_value, value_target):
    explained_variance = -1
    if th.var(state_value) > 0:
        explained_variance = max(
            -1, 1 - (th.var(value_target - state_value) / th.var(value_target))
        )
    return explained_variance
