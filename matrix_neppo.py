"""NePPO (Algorithm 1) on the two-player matrix games of eq. (7) with closed-form inner solvers.

This is the script behind Table 1 and Fig. 2 (Sec. 4.2). It is pure numpy: no JAX, torch or
jaxtrl. Player 1 mixes over rows A1 / A2 with P(A2) = p and player 2 over columns B1 / B2 with
P(B2) = q; the game is parameterized by alpha in [0, 1]:

    payoff_A = [[1,             1 - 2 alpha],       payoff_B = [[1 - 2 alpha,     (alpha + 1) / 2],
                [(1 - 3 alpha) / 2, alpha    ]]                  [(7 - 3 alpha) / 4, 2 - 3 alpha  ]]

The potential is Phi_w(p, q) = -(p - p_phi)^2 - (q - q_phi)^2 with w = (p_phi, q_phi) in [0, 1]^2.
Because the 2x2 game is analytic, both inner solvers of Algorithm 1 are exact:

  * CoopGameSolver (lines 6-7): the maximizer of the strictly concave Phi_w is pi^Phi = w
    projected onto [0, 1]^2 (`pm_step`).
  * RLSolver (lines 9-10): each player's exact pure best response to the other player's pi^Phi
    strategy (`br_player1`, `br_player2`).
  * F_i (line 13, eq. 3): [Phi_w(pi^Phi) - Phi_w(br_i, pi^Phi_-i)] - [J_i(pi^Phi) - J_i(br_i, pi^Phi_-i)]
    (`compute_F`), aggregated as F_beta = (1/beta) log sum_i exp(beta F_i) with beta = 1e4 (lines 15-16).
  * Line 17: a two-point zeroth-order step, u uniform on the unit sphere and dim(w) = 2, averaged
    over --n_samples directions, followed by projection of w onto [0, 1]^2 (`zo_gradient_step`).

Regret of a player is its best-response gain against the other player's strategy; Table 1 reports
the maximum over the two players.
"""

import numpy as np

# =============================================================================
# Core Functions
# =============================================================================


def phi(p, q, p_phi, q_phi):
    """Quadratic potential function: -(p - p_phi)^2 - (q - q_phi)^2"""
    return -((p - p_phi) ** 2) - ((q - q_phi) ** 2)


def reward_player1(p, q, alpha):
    """
    Player 1's expected reward.
    payoff_A[0,0]=1, [0,1]=1-2α, [1,0]=(1-3α)/2, [1,1]=α
    """
    return (
        (1 - p) * (1 - q) * 1
        + (1 - p) * q * (1 - 2 * alpha)
        + p * (1 - q) * (1 - 3 * alpha) / 2
        + p * q * alpha
    )


def reward_player2(p, q, alpha):
    """
    Player 2's expected reward.
    payoff_B[0,0]=1-2α, [0,1]=(α+1)/2, [1,0]=(7-3α)/4, [1,1]=2-3α
    """
    return (
        (1 - p) * (1 - q) * (1 - 2 * alpha)
        + (1 - p) * q * (alpha + 1) / 2
        + p * (1 - q) * (7 - 3 * alpha) / 4
        + p * q * (2 - 3 * alpha)
    )


def br_player1(q, alpha):
    """
    Analytical best response for Player 1.

    BR_1 based on: u_1(A2,q) - u_1(A1,q) = -(1+3α)/2 + (9α-1)/2 * q
    q* = (1+3α)/(9α-1)

    - If α > 1/9: BR_1(q) = 0 if q < q*, 1 if q > q*
    - If α = 1/9: BR_1(q) = 0 always
    - If α < 1/9: BR_1(q) = 1 if q < q*, 0 if q > q* (reversed since 9α-1 < 0)
    """
    if abs(9 * alpha - 1) < 1e-10:  # α ≈ 1/9
        return 0.0  # Always A1

    q_star = (1 + 3 * alpha) / (9 * alpha - 1)

    if alpha > 1 / 9:
        # Normal case: BR=A1 if q < q*, BR=A2 if q > q*
        return 0.0 if q < q_star else 1.0
    else:
        # Reversed (since 9α-1 < 0): BR=A2 if q < q*, BR=A1 if q > q*
        return 1.0 if q < q_star else 0.0


def br_player2(p, alpha):
    """
    Analytical best response for Player 2.

    BR_2 based on: u_2(p,B2) - u_2(p,B1) = (5α-1)/2 + (3-19α)/4 * p
    p* = 2(1-5α)/(3-19α)

    - If α < 3/19: BR_2(p) = 0 if p < p*, 1 if p > p*
    - If α = 3/19: BR_2(p) = 0 always
    - If α > 3/19: BR_2(p) = 1 if p < p*, 0 if p > p* (reversed since 3-19α < 0)
    """
    if abs(3 - 19 * alpha) < 1e-10:  # α ≈ 3/19
        return 0.0  # Always B1

    p_star = 2 * (1 - 5 * alpha) / (3 - 19 * alpha)

    if alpha < 3 / 19:
        # Normal case: BR=B1 if p < p*, BR=B2 if p > p*
        return 0.0 if p < p_star else 1.0
    else:
        # Reversed (since 3-19α < 0): BR=B2 if p < p*, BR=B1 if p > p*
        return 1.0 if p < p_star else 0.0


def pm_step(p_phi, q_phi, use_sigmoid=False):
    """PM step: maximize phi by outputting projected strategies.

    If use_sigmoid=True, uses sigmoid(phi) instead of clip(phi, 0, 1).
    """
    if use_sigmoid:
        return 1 / (1 + np.exp(-p_phi)), 1 / (1 + np.exp(-q_phi))
    else:
        return np.clip(p_phi, 0, 1), np.clip(q_phi, 0, 1)


def player_regrets(p, q, alpha):
    """Best-response gain of each player against the other's strategy (the regret of Table 1)."""
    r1 = max(0.0, reward_player1(br_player1(q, alpha), q, alpha) - reward_player1(p, q, alpha))
    r2 = max(0.0, reward_player2(p, br_player2(p, alpha), alpha) - reward_player2(p, q, alpha))
    return r1, r2


# =============================================================================
# F Computation
# =============================================================================


def compute_F(p_pm, q_pm, p_br1, q_br2, p_phi, q_phi, alpha):
    """
    Compute F values for both players.
    F_i = (Phi_PM - Phi_BR_i) - (V_i_PM - V_i_BR_i)
    """
    # Phi values
    Phi_pm = phi(p_pm, q_pm, p_phi, q_phi)
    Phi_br1 = phi(p_br1, q_pm, p_phi, q_phi)  # Player 1 deviates, player 2 stays
    Phi_br2 = phi(p_pm, q_br2, p_phi, q_phi)  # Player 2 deviates, player 1 stays

    # Reward values
    V1_pm = reward_player1(p_pm, q_pm, alpha)
    V1_br = reward_player1(p_br1, q_pm, alpha)
    V2_pm = reward_player2(p_pm, q_pm, alpha)
    V2_br = reward_player2(p_pm, q_br2, alpha)

    # F values
    F1 = (Phi_pm - Phi_br1) - (V1_pm - V1_br)
    F2 = (Phi_pm - Phi_br2) - (V2_pm - V2_br)

    return F1, F2


# =============================================================================
# Zeroth-Order Gradient Update
# =============================================================================


def zo_gradient_step(p_phi, q_phi, alpha, delta=1e-3, lr=1e-2, n_samples=1):
    """
    Perform a zeroth-order gradient step to minimize LSE(F1, F2).
    Uses n_samples random directions and averages the gradient estimates.
    """
    grad_accum = np.zeros(2)

    for _ in range(n_samples):
        # Sample random direction on unit sphere
        u = np.random.randn(2)
        u = u / np.linalg.norm(u)

        # Perturb phi in both directions
        phi_plus = (p_phi + delta * u[0], q_phi + delta * u[1])
        phi_minus = (p_phi - delta * u[0], q_phi - delta * u[1])

        # Run PM + BR + compute F for positive perturbation
        p_pm_p, q_pm_p = pm_step(*phi_plus)
        F1_p, F2_p = compute_F(
            p_pm_p,
            q_pm_p,
            br_player1(q_pm_p, alpha),
            br_player2(p_pm_p, alpha),
            *phi_plus,
            alpha,
        )

        # Run PM + BR + compute F for negative perturbation
        p_pm_m, q_pm_m = pm_step(*phi_minus)
        F1_m, F2_m = compute_F(
            p_pm_m,
            q_pm_m,
            br_player1(q_pm_m, alpha),
            br_player2(p_pm_m, alpha),
            *phi_minus,
            alpha,
        )

        BETA = 10_000
        F1_p = F1_p * BETA
        F2_p = F2_p * BETA
        F1_m = F1_m * BETA
        F2_m = F2_m * BETA

        # Log-sum-exp aggregation
        lse_plus = np.logaddexp(F1_p, F2_p) / BETA
        lse_minus = np.logaddexp(F1_m, F2_m) / BETA

        # Gradient estimate for this sample
        grad_scale = (2 / (2 * delta)) * (lse_plus - lse_minus)
        grad_accum += grad_scale * u

    # Average gradient over samples
    grad = grad_accum / n_samples

    # SGD update (minimize LSE of F)
    p_phi_new = p_phi - lr * grad[0]
    q_phi_new = q_phi - lr * grad[1]

    return np.clip(p_phi_new, 0, 1), np.clip(q_phi_new, 0, 1)


# =============================================================================
# Main Training Loop
# =============================================================================


def train(
    alpha=0.0,
    num_iters=500,
    delta=1e-3,
    lr=1e-2,
    lr_decay="none",
    delta_decay=False,
    n_samples=1,
    start_p=0.75,
    start_q=0.75,
    seed=42,
    verbose=True,
):
    """
    Run Algorithm 1 with exact inner solvers on the alpha game.

    Args:
        alpha: Game parameter (alpha=0 gives Nash at (0,0))
        num_iters: Number of outer iterations
        delta: Perturbation size for zeroth-order gradient
        lr: Initial learning rate
        lr_decay: Learning rate decay schedule ('none', 'linear', 'sqrt', '1_over_t', 'sqrt_slow', 'log', 'cosine')
        delta_decay: Decay delta as delta * (t+1)^(-0.1)
        n_samples: Number of ZO gradient samples per step (reduces variance)
        start_p, start_q: Initial w = (p_phi, q_phi)
        seed: Random seed

    Returns:
        history: Dictionary with training history
    """
    np.random.seed(seed)
    p_phi, q_phi = start_p, start_q  # Initialize at (start_p, start_q)

    history = {
        "p_phi": [],
        "q_phi": [],
        "p_pm": [],
        "q_pm": [],
        "nash_gap": [],
        "regret_1": [],
        "regret_2": [],
        "max_regret": [],
        "F1": [],
        "F2": [],
        "lr": [],
    }

    for t in range(num_iters):
        # Compute current learning rate based on decay schedule
        if lr_decay == "none":
            lr_t = lr
        elif lr_decay == "linear":
            # Linear decay from lr to 0
            lr_t = lr * (1 - t / num_iters)
        elif lr_decay == "sqrt":
            # 1/sqrt(t+1) decay
            lr_t = lr / np.sqrt(t + 1)
        elif lr_decay == "1_over_t":
            # 1/t decay (starting at t=1)
            lr_t = lr / (t + 1)
        elif lr_decay == "sqrt_slow":
            # Slower sqrt decay: 1/sqrt(t/100 + 1)
            lr_t = lr / np.sqrt(t / 100 + 1)
        elif lr_decay == "log":
            # Very slow log decay: 1/log(t+2)
            lr_t = lr / np.log(t + 2)
        elif lr_decay == "cosine":
            # Cosine annealing
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / num_iters))
        else:
            lr_t = lr

        # PM step
        p_pm, q_pm = pm_step(p_phi, q_phi)

        # Best responses
        p_br1 = br_player1(q_pm, alpha)
        q_br2 = br_player2(p_pm, alpha)

        # Compute F values
        F1, F2 = compute_F(p_pm, q_pm, p_br1, q_br2, p_phi, q_phi, alpha)

        # Regret of each player and the Nash gap (their sum)
        regret_1, regret_2 = player_regrets(p_pm, q_pm, alpha)
        nash_gap = regret_1 + regret_2

        # Record history
        history["p_phi"].append(p_phi)
        history["q_phi"].append(q_phi)
        history["p_pm"].append(p_pm)
        history["q_pm"].append(q_pm)
        history["nash_gap"].append(nash_gap)
        history["regret_1"].append(regret_1)
        history["regret_2"].append(regret_2)
        history["max_regret"].append(max(regret_1, regret_2))
        history["F1"].append(F1)
        history["F2"].append(F2)
        history["lr"].append(lr_t)

        # Compute delta with optional decay: delta_t = delta * (t+1)^(-0.1)
        if delta_decay:
            delta_t = delta * ((t + 1) ** (-0.1))
        else:
            delta_t = delta

        # ZO gradient update
        p_phi, q_phi = zo_gradient_step(p_phi, q_phi, alpha, delta_t, lr_t, n_samples)

        if verbose and t % 100 == 0:
            print(
                f"t={t}: phi=({p_phi:.4f}, {q_phi:.4f}), pm=({p_pm:.4f}, {q_pm:.4f}), "
                f"max_regret={max(regret_1, regret_2):.6f}, nash_gap={nash_gap:.6f}, lr={lr_t:.6f}"
            )

    return history


def write_history(history, path):
    """Write the per-iteration history as csv (e.g. for the regret curves of Fig. 2)."""
    import csv

    keys = list(history.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration"] + keys)
        for t in range(len(history[keys[0]])):
            w.writerow([t] + [history[k][t] for k in keys])


# =============================================================================
# Visualization
# =============================================================================


def plot_convergence(history, alpha, save_path=None):
    """Plot 4-panel convergence visualization."""
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Phi trajectory in (p, q) space
    ax = axes[0, 0]
    ax.plot(history["p_phi"], history["q_phi"], "b-", alpha=0.7, label="phi trajectory")
    ax.scatter(
        [history["p_phi"][0]],
        [history["q_phi"][0]],
        c="g",
        s=100,
        marker="o",
        label="start",
    )
    ax.scatter(
        [history["p_phi"][-1]],
        [history["q_phi"][-1]],
        c="r",
        s=100,
        marker="x",
        label="end",
    )
    ax.set_xlabel("p_phi")
    ax.set_ylabel("q_phi")
    ax.set_title(f"Phi Parameter Trajectory (alpha={alpha})")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: PM strategies over time
    ax = axes[0, 1]
    ax.plot(history["p_pm"], label="p (player 1)", alpha=0.7)
    ax.plot(history["q_pm"], label="q (player 2)", alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Strategy")
    ax.set_title("PM Strategies Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Max player regret over time (Fig. 2)
    ax = axes[1, 0]
    ax.plot(np.maximum(history["max_regret"], 1e-3), "r-", alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max Player Regret")
    ax.set_title("Max Player Regret Over Time")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Plot 4: F values over time
    ax = axes[1, 1]
    ax.plot(history["F1"], label="F1 (player 1)", alpha=0.7)
    ax.plot(history["F2"], label="F2 (player 2)", alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("F value")
    ax.set_title("Potential Difference (F) Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NePPO with exact inner solvers on the alpha games (Table 1, Fig. 2)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Game parameter alpha in [0, 1]")
    parser.add_argument(
        "--num_iters", type=int, default=500, help="Number of outer iterations (500 in Fig. 2)"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate eta")
    parser.add_argument(
        "--lr_decay",
        type=str,
        default="sqrt",
        choices=["none", "linear", "sqrt", "1_over_t", "sqrt_slow", "log", "cosine"],
        help="LR decay schedule",
    )
    parser.add_argument(
        "--delta", type=float, default=1e-2, help="ZO perturbation size delta"
    )
    parser.add_argument(
        "--delta_decay", action="store_true", help="decay delta as delta * (t+1)^(-0.1)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="Number of ZO gradient samples per step"
    )
    parser.add_argument("--start_p", type=float, default=0.75, help="initial p_phi")
    parser.add_argument("--start_q", type=float, default=0.75, help="initial q_phi")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--history", type=str, default=None, help="write the per-iteration history (csv)")
    parser.add_argument("--plot", type=str, default=None, help="save the convergence plot (png)")
    parser.add_argument("--show", action="store_true", help="show the convergence plot interactively")
    args = parser.parse_args()

    print(f"Running NePPO (exact inner solvers) on the alpha game with alpha={args.alpha}")
    print(
        f"Parameters: num_iters={args.num_iters}, lr={args.lr}, lr_decay={args.lr_decay}, delta={args.delta}, "
        f"delta_decay={args.delta_decay}, n_samples={args.n_samples}, start=({args.start_p}, {args.start_q}), seed={args.seed}"
    )
    print("-" * 60)

    history = train(
        alpha=args.alpha,
        num_iters=args.num_iters,
        delta=args.delta,
        lr=args.lr,
        lr_decay=args.lr_decay,
        delta_decay=args.delta_decay,
        n_samples=args.n_samples,
        start_p=args.start_p,
        start_q=args.start_q,
        seed=args.seed,
    )

    final_p_phi = history["p_phi"][-1]
    final_q_phi = history["q_phi"][-1]
    regs = np.array(history["max_regret"])

    print("-" * 60)
    print(f"Final: w = (p_phi, q_phi) = ({final_p_phi:.4f}, {final_q_phi:.4f}), pi^Phi = ({history['p_pm'][-1]:.4f}, {history['q_pm'][-1]:.4f})")
    print(f"Final player regrets: ({history['regret_1'][-1]:.4f}, {history['regret_2'][-1]:.4f})")
    print(f"Max player regret: final={regs[-1]:.4f}, mean over the last 10 iterations={regs[-10:].mean():.4f}, min={regs.min():.4f}")

    if args.history:
        write_history(history, args.history)
        print(f"wrote {args.history}")
    if args.plot or args.show:
        plot_convergence(history, args.alpha, save_path=args.plot)
        if args.plot:
            print(f"wrote {args.plot}")
