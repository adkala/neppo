import jax
import torch as th

#
from jaxtrl.rl.ppo.trainer import ACTOR_MAP as PPO_ACTOR_MAP
from jaxtrl.rl.ppo.trainer import CRITIC_FACTORY_MAP as PPO_CRITIC_FACTORY_MAP

from marl.algorithms.neppo.mixer import (
    NPConvexNN,
    NPLambda,
    NPMixer,
    NPMixerWithDelta,
    NPNoMix,
)
from marl.algorithms.neppo.neppo import NePPO


class NePPORunner:
    def __init__(self, agents, config, key, device=None):
        if device is not None:
            pass
        elif "device" in config:
            device = config["device"]
        else:
            device = "cuda" if th.cuda.is_available() else "cpu"

        algo_kwargs = config["algo_kwargs"]
        actor_kwargs = config["actor_kwargs"]
        critic_kwargs = config["critic_kwargs"]

        keys = jax.random.split(key, len(agents) + 1)
        actors = []
        for agent, key in zip(agents, keys[:-1]):
            actor = PPO_ACTOR_MAP[config["actor"]](
                agent.pi_obs_spec,
                agent.pi_action_spec,
                key,
                **{**actor_kwargs, "device": device},
            )
            actors.append(actor)

        PLACEHOLDER = agents[0].pi_obs_spec

        pm_critic = PPO_CRITIC_FACTORY_MAP[config["critic"]](
            PLACEHOLDER,
            None,
            **{
                **critic_kwargs,
                "device": device,
            },
        )

        br_critics = []
        for agent in agents:
            br_critic = PPO_CRITIC_FACTORY_MAP[config["critic"]](
                PLACEHOLDER,
                agent.pi_action_spec,
                **{
                    **critic_kwargs,
                    "device": device,
                },
            )
            br_critics.append(br_critic)

        if algo_kwargs.get("convex_nn", False):
            phi = NPConvexNN(
                input_dim=algo_kwargs["mixer_input_dim"],
                output_dim=algo_kwargs["mixer_output_dim"],
                hidden_size=algo_kwargs.get("convex_nn_hidden_size", 16),
            )
        elif algo_kwargs.get("nonmix", False):
            phi = NPNoMix(
                input_dim=algo_kwargs["mixer_input_dim"],
                hidden_size=algo_kwargs.get("nonmix_hidden_size", 64),
            )
        elif algo_kwargs["mixer"]:
            if algo_kwargs.get("mixer_with_delta", False):
                phi = NPMixerWithDelta(
                    input_dim=algo_kwargs["mixer_input_dim"],
                    output_dim=algo_kwargs["mixer_output_dim"],
                )
            else:
                phi = NPMixer(
                    input_dim=algo_kwargs["mixer_input_dim"],
                    output_dim=algo_kwargs["mixer_output_dim"],
                )
        else:
            phi = NPLambda(
                output_dim=algo_kwargs["mixer_output_dim"],
            )

        phi = phi.to(device)

        self.algo = NePPO(actors, pm_critic, br_critics, phi, device, **algo_kwargs)

    def step(self, tds):
        return self.algo.step(tds)

    @property
    def policies(self):
        return self.algo.current_policies

    @property
    def num_frames(self):
        return self.algo.num_frames

    def get_policy_contexts(self, key):
        return [
            policy.init_context(key)
            for policy, key in zip(
                self.policies, jax.random.split(key, len(self.policies))
            )
        ]
        return [
            policy.init_context(key)
            for policy, key in zip(
                self.policies, jax.random.split(key, len(self.policies))
            )
        ]
