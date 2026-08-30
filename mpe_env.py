from typing import Any

import gymnasium as gym
import jax
import numpy as np
from flax import struct
from jax import numpy as jnp
from jaxmarl import make

# --- JaxMARL Observation Wrapper ---


@struct.dataclass
class JaxMARLObs:
    """Observation wrapper that holds JaxMARL state for the collect loop."""

    obs_dict: dict  # agent_name -> obs array
    state: Any  # opaque JaxMARL state
    reward_dict: dict  # agent_name -> reward (from step)
    done_dict: dict  # agent_name -> done
    timestep: int
    key: jax.Array  # for next step call


# --- JaxMARL Agent ---


class JaxMARLAgent:
    """Minimal agent that provides specs and obs/action transforms."""

    def __init__(self, name, idx, obs_space, action_space):
        self.name = name
        self.idx = idx
        self._obs_space = obs_space
        self._action_space = action_space

    @property
    def pi_obs_spec(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._obs_space.shape, dtype=np.float32
        )

    @property
    def pi_action_spec(self):
        # Map from tanh output [-1, 1] to env action space [0, 1]
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._action_space.shape, dtype=np.float32
        )

    def obs_env2pi(self, obs: JaxMARLObs, ctx=None):
        return obs.obs_dict[self.name], ctx

    def action_pi2env(self, action, ctx=None):
        # Map from [-1, 1] to [0, 1] for MPE
        action_env = (action + 1.0) / 2.0
        return action_env, ctx

    def init_context(self, key):
        return None


# --- JaxMARL Reward ---


class JaxMARLReward:
    """Passthrough since JaxMARL already computes rewards."""

    def __init__(self, name):
        self.name = name

    def __call__(self, obs, action, next_obs, context=None):
        return next_obs.reward_dict[self.name], context

    def init_context(self, key):
        return None


# --- JaxMARL Sim ---


class JaxMARLSim:
    """Wrapper that adapts JaxMARL to the Sim interface."""

    def __init__(self, env, agents, rewards, max_steps):
        self.env = env
        self.agents = agents
        self.rewards = rewards
        self.max_steps = max_steps
        self.agent_names = env.agents  # ordered list

    def reset(self, key):
        key, key_reset = jax.random.split(key)
        obs_dict, state = self.env.reset(key_reset)

        # Create zero reward/done dicts for initial obs
        reward_dict = {n: jnp.array(0.0) for n in self.agent_names}
        done_dict = {n: jnp.array(False) for n in self.agent_names}
        done_dict["__all__"] = jnp.array(False)

        raw_obs = JaxMARLObs(
            obs_dict=obs_dict,
            state=state,
            reward_dict=reward_dict,
            done_dict=done_dict,
            timestep=0,
            key=key,
        )

        obs_for_pi = [obs_dict[a.name] for a in self.agents]
        agent_ctx = [None] * len(self.agents)
        sim_ctx = [None] * len(self.rewards)

        return raw_obs, obs_for_pi, agent_ctx, sim_ctx

    def step(self, obs, actions_from_pi, agent_ctx, sim_ctx):
        key, step_key = jax.random.split(obs.key)

        # Convert list -> dict, applying action transform
        actions = {}
        for i, agent in enumerate(self.agents):
            action_env, _ = agent.action_pi2env(actions_from_pi[i], agent_ctx[i])
            actions[agent.name] = action_env

        obs_dict, state, reward_dict, done_dict, _ = self.env.step(
            step_key, obs.state, actions
        )

        next_obs = JaxMARLObs(
            obs_dict=obs_dict,
            state=state,
            reward_dict=reward_dict,
            done_dict=done_dict,
            timestep=obs.timestep + 1,
            key=key,
        )

        # Stack rewards into array (one per agent)
        reward = jnp.array([reward_dict[n] for n in self.agent_names])

        terminated = done_dict["__all__"]
        truncated = next_obs.timestep >= self.max_steps

        next_obs_for_pi = [obs_dict[a.name] for a in self.agents]

        return (
            next_obs,
            reward,
            terminated,
            truncated,
            next_obs_for_pi,
            agent_ctx,
            sim_ctx,
        )


# --- State Agent for Critic ---


class JaxMARLStateAgent:
    """Agent that extracts global state for the critic by concatenating all agent observations.

    This matches the MAPPO paper (Yu et al., 2022) which uses concatenated
    agent observations as the shared state for the centralized critic.
    Optionally includes action dimensions when include_actions=True.
    """

    def __init__(self, agents, include_actions=False):
        self.agents = agents
        self.include_actions = include_actions
        self._obs_dim = sum(a.pi_obs_spec.shape[0] for a in agents)
        self._action_dim = sum(a.pi_action_spec.shape[0] for a in agents) if include_actions else 0
        self._state_dim = self._obs_dim + self._action_dim

    def obs_env2pi(self, obs: JaxMARLObs, ctx=None):
        state = jnp.concatenate([obs.obs_dict[a.name] for a in self.agents])
        return state, ctx


class JaxMARLEnvStateAgent:
    """Agent that extracts the true environment state (positions + velocities)
    for the centralized critic, rather than concatenated observations.

    This provides a more compact, privileged state representation.
    """

    def __init__(self, env):
        key = jax.random.key(0)
        _, state = env.reset(key)
        self._state_dim = state.p_pos.size + state.p_vel.size

    def obs_env2pi(self, obs: JaxMARLObs, ctx=None):
        state = jnp.concatenate([obs.state.p_pos.flatten(), obs.state.p_vel.flatten()])
        return state, ctx


# --- Factory ---


def make_jaxmarl_env(
    env_name, action_type="Continuous", use_env_state=False, include_actions=False, **env_kwargs
):
    """Create JaxMARL environment and associated agents/rewards.

    Args:
        env_name: JaxMARL environment name (e.g. "MPE_simple_world_comm_v3").
        action_type: Action type ("Continuous" or "Discrete").
        use_env_state: If True, use true environment state (positions + velocities)
            for the centralized critic instead of concatenated agent observations.
        include_actions: If True, include action dimensions in the state for the
            centralized critic. Actions are concatenated by the caller.
    """
    env = make(env_name, action_type=action_type, **env_kwargs)

    agents = [
        JaxMARLAgent(name, i, env.observation_space(name), env.action_space(name))
        for i, name in enumerate(env.agents)
    ]
    rewards = [JaxMARLReward(name) for name in env.agents]

    if use_env_state:
        state_agent = JaxMARLEnvStateAgent(env)
    else:
        state_agent = JaxMARLStateAgent(agents, include_actions=include_actions)

    return env, agents, rewards, state_agent
