# NePPO: Near-Potential Policy Optimization

Code for **"NePPO: Near-Potential Policy Optimization for General-Sum Multi-Agent
Reinforcement Learning"** ([arXiv:2603.06977](https://arxiv.org/abs/2603.06977)).

NePPO learns a player-independent potential function Φ such that a Nash equilibrium of the
cooperative game with Φ as the common utility is an approximate Nash equilibrium of the
original general-sum game. Each iteration of Algorithm 1 (i) solves the cooperative game
induced by Φ (`CoopGameSolver`), (ii) computes each player's best response to that joint
policy (`RLSolver`), and (iii) updates the parameters of Φ with a two-point zeroth-order
gradient of the smoothed objective `F̄_β(Φ) = (1/β) log Σ_i exp(β F_i(Φ))`.

## Setup

Install `torch` and `jax` builds matching your platform (CPU / CUDA) first, then

```bash
pip install -e .
```

Requires Python ≥ 3.10. Rollout collection and PPO / IPPO / MAPPO training come from
[jaxtrl](https://github.com/adkala/jaxtrl) (JAX-jitted env stepping, TorchRL updates), which
`pip` pulls from GitHub pinned to the commit used for the paper. Other core dependencies:
`jaxmarl`, `torchrl`, `tensordict`, `flax`. `matrix_neppo.py` needs only `numpy` (`pip install -e .[plots]` adds
`matplotlib` for `--plot`). Training logs go to Weights & Biases; set `WANDB_MODE=offline`
(or `disabled`) to run without an account. The RL scripts accept `--test` for a fast CPU
smoke run, `--device cpu|cuda`, and `--num_env_steps`.

## Repository layout

```
matrix_game_env.py          The alpha-parameterized matrix game family for the JaxMARL-based scripts
matrix_neppo.py             NePPO on the 2x2 matrix games with closed-form inner solvers
matrix_neppo_ppo.py         NePPO on the matrix games with HAPPO / PPO inner solvers
matrix_ippo.py, matrix_mappo.py   IPPO / MAPPO baselines on the matrix games
matrix_regret.py            Closed-form regret of the strategies logged by matrix_{neppo_ppo,ippo,mappo}.py
mpe_env.py                  JaxMARL environment adapter for jaxtrl (MPE; also used by the matrix scripts)
mpe_neppo.py                NePPO on MPE Simple World Comm
mpe_ippo.py, mpe_mappo.py, mpe_maddpg.py   IPPO / MAPPO / MADDPG baselines on MPE
mpe_regret.py               Regret evaluation on MPE
marl/algorithms/neppo/      NePPO implementation (HAPPO cooperative step, PPO best-response step, zeroth-order Φ update)
marl/algorithms/neppo/mixer.py      Potential parameterizations (NPLambda, NPMixer, ...)
```

## Matrix games (Sec. 4.1 and 4.2)

The games of eq. (7) are indexed by `alpha ∈ [0, 1]` (`alpha = 0` is the game of eq. 8,
`alpha = 1` is matching pennies). A player's mixed strategy is the probability of its second
action, and its regret is its best-response gain against the other player's strategy; the
tables report the maximum over the two players.

**Sec. 4.1** runs Algorithm 1 with HAPPO as CoopGameSolver and PPO as RLSolver
(`marl/algorithms/neppo/`, the implementation also used on Simple World Comm) with the
potential `Φ_w = w J_1 + (1 - w) J_2` on the `alpha = 0` game (`w` is logged as `phi/w`):

```bash
python matrix_neppo_ppo.py --phi convex --alpha 0
python matrix_mappo.py --alpha 0          # MAPPO comparison (Fig. 1d)
```

**Sec. 4.2** uses the quadratic potential
`Φ_w(x, y) = -(x - p)^2 - (y - q)^2`, `w = (p, q) ∈ [0, 1]^2`:

```bash
for a in 0 0.2 0.4 0.6 0.8 1; do
  python matrix_neppo.py --alpha $a    # NePPO baseline
  python matrix_ippo.py  --alpha $a    # IPPO baseline
  python matrix_mappo.py --alpha $a    # MAPPO baseline
done
```

For a 2x2 game, both inner solvers of Algorithm 1 have closed forms. The maximizer of the
strictly concave `Φ_w` is `π^Φ = (p, q)` and each best response is a pure action. These are
used in `matrix_neppo.py`. A sampled CoopGameSolver differs from this by at most
≈ `2δ`: HAPPO's policy is the solution of the cooperative game under the perturbed potential
`w ± δu`, plus a small offset from the entropy bonus at the pure equilibria.
We verified that the sampled pipeline reaches the same equilibria for every `alpha`, 
with regrets within ≈ `2δ` of the closed-form values.

## Simple World Comm (Sec. 4.3)

The potential is the state-dependent mixture of eq. (9), `φ_w(s, a) = Σ_i softmax(W s + b)_i r_i(s, a)`
(`NPMixer`). Train NePPO and the baselines with

```bash
python mpe_neppo.py  --env MPE_simple_world_comm_v3
python mpe_ippo.py   --env MPE_simple_world_comm_v3
python mpe_mappo.py  --env MPE_simple_world_comm_v3
python mpe_maddpg.py --env MPE_simple_world_comm_v3
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{kalanther2026neppo,
  author       = {Kalanther, Addison and Bharvirkar, Sanika and Sastry, Shankar and Maheshwari, Chinmay},
  title        = {NePPO: Near-Potential Policy Optimization for General-Sum Multi-Agent Reinforcement Learning},
  booktitle    = {Proceedings of the 2026 65th IEEE Conference on Decision and Control (CDC)},
  year         = {2026},
  month        = {Dec},
  organization = {IEEE},
  note         = {To appear}
}
```
