import jax
import torch as th
from tensordict import TensorDict


def transitions_to_tensordicts(transitions, info_parser=None):
    tds = []
    for i in range(len(transitions.obs)):
        transition = transitions[i]

        obs = tdl(transition.obs)
        action = tdl(transition.action)
        step_count = tdl(transition.step_count)

        next_obs = tdl(transition.next_obs)
        reward = tdl(transition.reward)
        terminated = tdl(transition.terminated)
        truncated = tdl(transition.truncated)
        episode_reward = tdl(transition.episode_reward)

        td = TensorDict(
            source={
                "observation": obs,
                "action": action,
                "next": {
                    "observation": next_obs,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": terminated | truncated,
                    "step_count": step_count + 1,
                    "episode_reward": episode_reward,
                },
                "step_count": step_count,
            },
            batch_size=obs.shape[0],
            device=obs.device,
        )

        tds.append(td)

    return tds


def tdl(array):
    return th.from_dlpack(array)
