"""Restart-budget curves from the stored P(q) ensembles.

Exact subsampling-without-replacement estimator of E[max of B restarts]
over the stored 200-restart batches, bootstrap error bands, log-budget
slopes, and the two registered model slopes (anchored iid-Exp; anchored
LN x Exp mixture with the constants frozen in 05-envelope).

Zero new optimization: this is a reanalysis of results/pq/*.npz.

Usage:
    python analysis/budget_scan.py     # ~10 s, writes budget_scan.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import gammaln  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from envelope_prediction import (  # noqa: E402  (frozen constants of 05)
    ANCHOR_DELTA_PT,
    ANCHOR_N,
    SIGMA2,
    parameters,
    quantile,
)

QUBIT_COUNTS = (8, 10, 12, 14)
INSTANCES = (0, 1, 2)
BUDGETS = (6, 12, 25, 50, 100, 200)
SLOPE_BUDGETS = (25, 50, 100, 200)
BOOTSTRAP = 400
RESULTS = Path(__file__).resolve().parent.parent / "results" / "pq"


def log_choose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def expected_max(values: np.ndarray, budget: int) -> float:
    """Exact E[max of `budget` draws without replacement] from the sample."""
    ordered = np.sort(values)
    m = len(ordered)
    ranks = np.arange(1, m + 1)
    with np.errstate(divide="ignore"):
        log_weights = np.where(
            ranks >= budget,
            log_choose(ranks - 1, budget - 1) - log_choose(m, budget),
            -np.inf,
        )
    weights = np.exp(log_weights)
    weights /= weights.sum()  # exact up to float roundoff
    return float(np.dot(weights, ordered))


def slope_vs_logb(values: np.ndarray, budgets) -> float:
    curve = [expected_max(values, b) for b in budgets]
    return float(np.polyfit(np.log(budgets), curve, 1)[0])


def self_tests() -> None:
    rng = np.random.default_rng(3)
    sample = rng.exponential(size=200)
    assert abs(expected_max(sample, 200) - sample.max()) < 1e-12
    assert abs(expected_max(sample, 1) - sample.mean()) < 1e-12
    # Monte-Carlo agreement at an intermediate budget.
    draws = np.array(
        [rng.choice(sample, size=25, replace=False).max() for _ in range(20000)]
    )
    exact = expected_max(sample, 25)
    stderr = draws.std() / np.sqrt(len(draws))
    assert abs(exact - draws.mean()) < 4 * stderr, (exact, draws.mean(), stderr)
    print(f"self-tests passed (B=200 -> max, B=1 -> mean, MC check "
          f"{exact:.4f} vs {draws.mean():.4f} +/- {stderr:.4f})")


def model_slopes(num_qubits: int) -> tuple[float, float]:
    dim = 2.0**num_qubits
    iid_exp = 1.0 / dim
    c2 = ANCHOR_DELTA_PT * 2.0**ANCHOR_N / parameters(ANCHOR_N)
    log_n = c2 * parameters(num_qubits)
    v_hi = quantile(log_n + 0.5, SIGMA2[num_qubits])
    v_lo = quantile(log_n - 0.5, SIGMA2[num_qubits])
    return iid_exp, float((v_hi - v_lo) / dim)


def main() -> None:
    self_tests()
    fig, axes = plt.subplots(1, len(QUBIT_COUNTS), figsize=(13, 3.1), sharex=True)
    print(
        f"\n{'n':>4} {'inst':>4} {'slope/e-fold':>13} {'P1 iid-Exp':>11} "
        f"{'P2 LNxExp':>10} {'ratio/P1':>9}"
    )
    pooled = {}
    for axis, num_qubits in zip(axes, QUBIT_COUNTS, strict=True):
        iid_slope, mixture_slope = model_slopes(num_qubits)
        slopes = []
        for instance in INSTANCES:
            data = np.load(RESULTS / f"pq_n{num_qubits}_i{instance}_sigma0.1.npz")
            values = data["peak_weights"]
            curve = np.array([expected_max(values, b) for b in BUDGETS])
            slope = slope_vs_logb(values, SLOPE_BUDGETS)
            slopes.append(slope)

            boot = np.random.default_rng(instance + 10 * num_qubits)
            resampled = np.empty((BOOTSTRAP, len(BUDGETS)))
            for row in range(BOOTSTRAP):
                pick = values[boot.integers(len(values), size=len(values))]
                resampled[row] = [expected_max(pick, b) for b in BUDGETS]
            lo, hi = np.percentile(resampled, [16, 84], axis=0)

            axis.fill_between(BUDGETS, lo, hi, alpha=0.2)
            axis.plot(BUDGETS, curve, "o-", markersize=3.5,
                      label=f"instance {instance}")
            print(
                f"{num_qubits:>4} {instance:>4} {slope:>13.2e} "
                f"{iid_slope:>11.1e} {mixture_slope:>10.1e} "
                f"{slope / iid_slope:>9.1f}"
            )
        pooled[num_qubits] = (float(np.mean(slopes)), iid_slope, mixture_slope)
        axis.set_xscale("log")
        axis.set_title(f"$n = {num_qubits}$", fontsize=10)
        axis.set_xlabel("budget $B$")
        if num_qubits == QUBIT_COUNTS[0]:
            axis.set_ylabel(r"$\mathbb{E}[\max_B \delta]$")
            axis.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Budget curves from stored ensembles (bands: bootstrap 68%)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).resolve().parent / "budget_scan.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}\n")

    print("pooled slope per e-fold of budget vs registered models:")
    for num_qubits, (slope, iid_slope, mixture_slope) in pooled.items():
        print(
            f"  n = {num_qubits}: measured {slope:.2e}  "
            f"P1 {iid_slope:.1e} (x{slope / iid_slope:.0f})  "
            f"P2 {mixture_slope:.1e}"
        )


if __name__ == "__main__":
    main()
