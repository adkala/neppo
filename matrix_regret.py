"""Regret evaluation for the 2x2 matrix games (Sec. 4.2, Table 1, Fig. 2).

The matrix-game counterpart of mpe_regret.py. On Simple World Comm the best response has
to be trained (mpe_regret.py runs PPO against the other agents' frozen actors); on a 2x2
game it is a pure action and the regret has a closed form. The matrix scripts log each
player's mixed strategy -- the probability of its second action -- as
``strategies/<prefix>player_{A,B}/mean``: ``strategies/pm/...`` is the policy pi^Phi
returned by matrix_neppo_ppo.py, ``strategies/...`` the policy of matrix_ippo.py and
matrix_mappo.py. For a strategy profile (x, y) of the game with parameter alpha
(matrix_game_env.AlphaGameEnv)

    regret_A = max_a J_A(a, y) - J_A(x, y)
    regret_B = max_b J_B(x, b) - J_B(x, y)

and Table 1 reports max(regret_A, regret_B).

    python matrix_regret.py --alpha 0.6 --x 0.35 --y 0.42     # one strategy profile
    python matrix_regret.py --run entity/project/run_id        # a W&B run's logged history
    python matrix_regret.py --history run.jsonl                # an exported history (jsonl / csv)
    python matrix_regret.py --run ... --stride 5 --offset 4 --last 50   # last HAPPO step of each
                                                               # potential update (K1 = 5), last 50

``--alpha`` is read from the run or file name (``alpha_0.6``, ``a0.6``) when omitted.
"""
import csv
import functools
import json
import os
import re

import numpy as np

from matrix_game_env import AlphaGameEnv

PLAYERS = ("player_A", "player_B")
STEP_KEYS = ("_step", "step", "Step", "time/total_frames")


@functools.lru_cache(maxsize=None)
def payoffs(alpha):
    """(payoff_A, payoff_B) of the alpha-game as numpy arrays; rows index player A's actions."""
    env = AlphaGameEnv(alpha)
    return np.asarray(env.payoff_A, dtype=np.float64), np.asarray(env.payoff_B, dtype=np.float64)


def regret(x, y, alpha):
    """Closed-form regret of the profile (x, y): probability of the second action for A and B.

    Returns (max_regret, regret_A, regret_B, J_A, J_B).
    """
    A, B = payoffs(float(alpha))
    pA = np.array([1.0 - x, x])
    pB = np.array([1.0 - y, y])
    J_A = pA @ A @ pB
    J_B = pA @ B @ pB
    r_A = (A @ pB).max() - J_A  # best pure action of A against y
    r_B = (pA @ B).max() - J_B  # best pure action of B against x
    return max(r_A, r_B), r_A, r_B, J_A, J_B


def strategy_keys(prefix):
    return tuple(f"strategies/{prefix}{p}/mean" for p in PLAYERS)


def detect_prefix(keys):
    """'pm/' for a NePPO log, '' for an IPPO / MAPPO log."""
    for prefix in ("pm/", ""):
        if all(k in keys for k in strategy_keys(prefix)):
            return prefix
    raise KeyError("no strategies/[pm/]player_{A,B}/mean columns in the history")


def regret_curve(records, alpha, prefix=None, stride=1, offset=0):
    """Per-record regret of the logged strategies.

    records: iterable of dicts, one per logged step (only those holding the strategy keys
    count); `stride` / `offset` keep every stride-th of them starting at `offset` (e.g.
    `stride=K1, offset=K1-1` for the last HAPPO step of each potential update of
    matrix_neppo_ppo.py). Returns (steps, x, y, regret) as numpy arrays.
    """
    records = list(records)
    if prefix is None:
        prefix = detect_prefix({k for r in records for k in r})
    kA, kB = strategy_keys(prefix)
    rows = [r for r in records if kA in r and kB in r][offset::stride]
    if not rows:
        raise KeyError(f"no records with {kA} and {kB}")
    steps = np.array(
        [next((r[k] for k in STEP_KEYS if k in r), i) for i, r in enumerate(rows)], dtype=np.float64
    )
    x = np.array([r[kA] for r in rows], dtype=np.float64)
    y = np.array([r[kB] for r in rows], dtype=np.float64)
    regs = np.array([regret(xi, yi, alpha)[0] for xi, yi in zip(x, y)])
    return steps, x, y, regs


def summarize(steps, x, y, regs, alpha, last=10):
    """Final profile (mean of the last `last` records), its regret, and run-level statistics."""
    x_f, y_f = x[-last:].mean(), y[-last:].mean()
    r_max, r_A, r_B, J_A, J_B = regret(x_f, y_f, alpha)
    tail = regs[int(0.8 * len(regs)):]
    return {
        "records": len(regs),
        "last_step": steps[-1],
        "x": x_f,
        "y": y_f,
        "regret": r_max,
        "regret_A": r_A,
        "regret_B": r_B,
        "J_A": J_A,
        "J_B": J_B,
        "regret_mean_last": regs[-last:].mean(),
        "min_regret": regs.min(),
        "max_regret_last20": tail.max(),
    }


def infer_alpha(name):
    m = re.search(r"alpha[_=]?([0-9]*\.?[0-9]+)", name) or re.search(r"(?:^|_)a([0-9]*\.?[0-9]+)_", name)
    if m is None:
        raise ValueError(f"cannot read alpha from {name!r}; pass --alpha")
    return float(m.group(1))


def load_history(path):
    """Records of a history exported as jsonl (one dict per line) or csv (one column per key)."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rec = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
            records.append(rec)
    return records


def wandb_history(run_path, prefix=None):
    """Full (unsampled) strategy history of a W&B run. Returns (run name, records, prefix)."""
    import wandb

    run = wandb.Api().run(run_path)
    for p in (prefix,) if prefix is not None else ("pm/", ""):
        records = list(run.scan_history(keys=[*strategy_keys(p), "_step"]))
        if records:
            return run.name, records, p
    raise KeyError(f"no strategies/[pm/]player_{{A,B}}/mean in the history of {run_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Closed-form regret of logged matrix-game strategies (Table 1)"
    )
    parser.add_argument("--alpha", type=float, default=None, help="game parameter in [0, 1]")
    parser.add_argument("--x", type=float, default=None, help="player A: P(second action)")
    parser.add_argument("--y", type=float, default=None, help="player B: P(second action)")
    parser.add_argument("--run", type=str, default=None, help="W&B run path entity/project/run_id")
    parser.add_argument("--history", type=str, default=None, help="exported history (.jsonl or .csv)")
    parser.add_argument(
        "--prefix", type=str, default=None, help="'pm/' (NePPO) or '' (IPPO/MAPPO); auto-detected"
    )
    parser.add_argument("--last", type=int, default=10, help="records averaged for the final profile")
    parser.add_argument(
        "--stride", type=int, default=1, help="keep every stride-th strategy record (see --offset)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="first record to keep; --stride 5 --offset 4 selects the last of the 5 HAPPO steps "
        "of each potential update of matrix_neppo_ppo.py",
    )
    parser.add_argument("--curve", type=str, default=None, help="write the per-record regret curve (csv)")
    args = parser.parse_args()

    if (args.x is None) != (args.y is None):
        parser.error("--x and --y go together")
    if sum(v is not None for v in (args.x, args.run, args.history)) != 1:
        parser.error("give exactly one of --x/--y, --run, --history")

    if args.x is not None:
        if args.alpha is None:
            parser.error("--alpha is required with --x/--y")
        r_max, r_A, r_B, J_A, J_B = regret(args.x, args.y, args.alpha)
        print(f"alpha={args.alpha} (x, y)=({args.x:.4f}, {args.y:.4f})  J_A={J_A:.4f} J_B={J_B:.4f}")
        print(f"regret: A {r_A:.4f}  B {r_B:.4f}  max {r_max:.4f}")
        raise SystemExit

    try:
        if args.run is not None:
            name, records, prefix = wandb_history(args.run, args.prefix)
        else:
            name, records, prefix = os.path.basename(args.history), load_history(args.history), args.prefix
            if prefix is None:
                prefix = detect_prefix({k for r in records for k in r})
        alpha = args.alpha if args.alpha is not None else infer_alpha(name)
        steps, x, y, regs = regret_curve(records, alpha, prefix, args.stride, args.offset)
    except (KeyError, ValueError) as e:
        parser.error(str(e.args[0] if e.args else e))
    s = summarize(steps, x, y, regs, alpha, last=args.last)

    print(
        f"{name}: alpha={alpha} {s['records']} records of strategies/{prefix}player_*"
        f"  last step {s['last_step']:,.0f}"
    )
    print(f"final (x, y)=({s['x']:.4f}, {s['y']:.4f})  [mean of last {args.last}]  J_A={s['J_A']:.4f} J_B={s['J_B']:.4f}")
    print(f"regret: A {s['regret_A']:.4f}  B {s['regret_B']:.4f}  max {s['regret']:.4f}")
    print(
        f"mean per-record max regret over the last {args.last}: {s['regret_mean_last']:.4f}"
        f"  over the run: min {s['min_regret']:.4f}  max over last 20% {s['max_regret_last20']:.4f}"
    )

    if args.curve is not None:
        with open(args.curve, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "x", "y", "regret"])
            w.writerows(zip(steps, x, y, regs))
        print(f"curve written to {args.curve}")
