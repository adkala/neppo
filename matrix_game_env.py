"""Two-player matrix games used in Section 4 of the paper.

Each episode is a single play of the game. Each player's policy outputs a
value in [0, 1] -- the probability of choosing its second action -- and the
environment returns the *expected* payoff under the resulting mixed-strategy
profile (no sampling), so the policy output is the mixed strategy itself.
"""

import gymnasium as gym
import numpy as np
from flax import struct
from jax import numpy as jnp

from mpe_env import JaxMARLAgent, JaxMARLReward, JaxMARLStateAgent


@struct.dataclass
class MatrixGameState:
    timestep: int


class MatrixGameEnv:
    """Generic 2x2 matrix game given row-player (A) and column-player (B) payoffs."""

    def __init__(self, payoff_A, payoff_B):
        self.payoff_A = payoff_A
        self.payoff_B = payoff_B
        self.agents = ["player_A", "player_B"]

    def observation_space(self, agent_name):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def action_space(self, agent_name):
        return gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, key):
        state = MatrixGameState(timestep=0)
        obs_dict = {
            "player_A": jnp.array([1.0]),
            "player_B": jnp.array([1.0]),
        }
        return obs_dict, state

    def step(self, key, state, actions):
        # Actions arrive in [0, 1] = probability of choosing the second action.
        pA2 = jnp.clip(actions["player_A"][0], 0.0, 1.0)
        pB2 = jnp.clip(actions["player_B"][0], 0.0, 1.0)

        pA = jnp.array([1.0 - pA2, pA2])
        pB = jnp.array([1.0 - pB2, pB2])

        r_A = pA @ self.payoff_A @ pB
        r_B = pA @ self.payoff_B @ pB

        new_state = MatrixGameState(timestep=state.timestep + 1)
        obs_dict = {
            "player_A": jnp.array([1.0]),
            "player_B": jnp.array([1.0]),
        }
        reward_dict = {"player_A": r_A, "player_B": r_B}
        done_dict = {
            "player_A": jnp.array(True),
            "player_B": jnp.array(True),
            "__all__": jnp.array(True),
        }
        return obs_dict, new_state, reward_dict, done_dict, {}


class AlphaGameEnv(MatrixGameEnv):
    """The class of games in eq. (7) of the paper, indexed by alpha in [0, 1].

              B1                        B2
    A1    (1, 1-2a)              (1-2a, (a+1)/2)
    A2    ((1-3a)/2, (7-3a)/4)   (a, 2-3a)

    alpha = 0 is the general-sum game of eq. (8) (pure Nash equilibrium (A1, B1));
    alpha = 1 is zero-sum (matching pennies).
    """

    def __init__(self, alpha=0.0):
        a = float(alpha)
        super().__init__(
            payoff_A=jnp.array([[1.0, 1.0 - 2.0 * a], [(1.0 - 3.0 * a) / 2.0, a]]),
            payoff_B=jnp.array(
                [
                    [1.0 - 2.0 * a, (a + 1.0) / 2.0],
                    [(7.0 - 3.0 * a) / 4.0, 2.0 - 3.0 * a],
                ]
            ),
        )


def make_matrix_game_env(alpha=0.0, payoff_A=None, payoff_B=None):
    """Build the alpha-game (or a custom 2x2 game if both payoff matrices are given)."""
    if payoff_A is not None and payoff_B is not None:
        env = MatrixGameEnv(payoff_A, payoff_B)
    else:
        env = AlphaGameEnv(alpha)
    agents = [
        JaxMARLAgent(name, i, env.observation_space(name), env.action_space(name))
        for i, name in enumerate(env.agents)
    ]
    rewards = [JaxMARLReward(name) for name in env.agents]
    state_agent = JaxMARLStateAgent(agents)
    return env, agents, rewards, state_agent
