"""Exact fixed-base test and the last-doubling decomposition (controls).

Two post-registration controls on the reach statistics.

1. The fixed-base p of the abstract, computed exactly. The weighted
   chi-square of converged_reach.py treats the instance-mean errors as
   known; its own self-test reports mild anti-conservatism. Here the
   same hypothesis, an affine law for ln R16 in n over n = 8..14, is
   tested with the covariance estimated: per-instance ln R16 forms
   eighteen i.i.d. 4-vectors, the two contrasts orthogonal to the
   affine design are tested by Hotelling's T^2, referred to F(2, 16).
   The registered branch decisions rest on the steepening statistic T,
   not on this p; only the quoted number changes.

2. The last budget doubling at n = 16, decomposed per instance. The
   acceptance criterion reads the grid mean (0.362%, committed in
   convergence_diagnostics.log); this control shows where that gain
   lives: nine tenths of it in a single instance.

Self-tests: the grid means reproduce the R16 columns of
converged_reach.log at both grids; a planted affine law is accepted
and a planted curved law rejected; the grid gain of section 2
reproduces the committed 0.362%.

Usage:
    python analysis/reach_exact_test.py        # ~1 min
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIZES = (8, 10, 12, 14)
INSTANCES = 18
B0 = 16


def expected_best(values, budget=B0):
    """E[max of `budget` draws without replacement] over the stored restarts."""
    ordered = np.sort(np.asarray(values, dtype=float))
    total = len(ordered)
    norm = comb(total, budget)
    return float(sum(v * (comb(k + 1, budget) - comb(k, budget))
                     for k, v in enumerate(ordered) if k >= budget - 1) / norm)


def load_r16(grid, n, instance):
    if grid == "frozen":
        path = RESULTS / "pq" / f"pq_n{n}_i{instance}_sigma0.1.npz"
    else:
        path = next((RESULTS / "converged").glob(f"*n{n}_i{instance}_*.npz"))
    return expected_best(np.load(path, allow_pickle=True)["peak_weights"])


def exact_test(log_r16):
    """Hotelling T^2 on the contrasts orthogonal to an affine law in n."""
    design = np.column_stack([np.ones(len(SIZES)), np.array(SIZES, float)])
    q, _ = np.linalg.qr(design, mode="complete")
    contrasts = q[:, 2:].T
    z = log_r16 @ contrasts.T
    z_bar = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    m = z.shape[0]
    t2 = m * z_bar @ np.linalg.solve(cov, z_bar)
    f_stat = t2 * (m - 2) / (2 * (m - 1))
    return t2, f_stat, float(f_dist.sf(f_stat, 2, m - 2))


def self_tests():
    rng = np.random.default_rng(11)
    x = np.array(SIZES, float)
    affine = 1.0 - 0.35 * x
    planted = affine + 0.02 * rng.standard_normal((INSTANCES, 4))
    _, _, p_affine = exact_test(planted)
    curved = affine + 0.08 * (x - 11.0) ** 2 / 10.0
    planted_curved = curved + 0.02 * rng.standard_normal((INSTANCES, 4))
    _, _, p_curved = exact_test(planted_curved)
    assert p_affine > 0.2 and p_curved < 1e-4, (p_affine, p_curved)
    print("self-tests passed (planted affine accepted, planted curve "
          "rejected)")


def section_one():
    print("\n1. Exact fixed-base test on n <= 14 (Hotelling, F(2,16))\n")
    reference = {"frozen": [0.6903, 0.5064, 0.3390, 0.2057],
                 "converged": [0.6909, 0.5106, 0.3534, 0.2202]}
    for grid in ("frozen", "converged"):
        r16 = np.array([[load_r16(grid, n, k) for n in SIZES]
                        for k in range(INSTANCES)])
        means = r16.mean(axis=0)
        assert np.allclose(means, reference[grid], atol=5e-4), (grid, means)
        t2, f_stat, p = exact_test(np.log(r16))
        print(f"   {grid:>9}: grid means reproduce converged_reach.log; "
              f"T^2 = {t2:.2f}, F(2,16) = {f_stat:.2f}, p = {p:.4f}")


def section_two():
    print("\n2. Last budget doubling at n = 16, per instance "
          "(6400 -> 12800 steps)\n")
    rows = []
    for path in sorted((RESULTS / "converged").glob("*n16*.npz")):
        data = np.load(path, allow_pickle=True)
        ladder = data["ladder_weights"].copy()
        for rung in range(1, ladder.shape[1]):
            stopped = np.isnan(ladder[:, rung])
            ladder[stopped, rung] = ladder[stopped, rung - 1]
        rows.append((int(data["instance_index"]),
                     expected_best(ladder[:, -2]),
                     expected_best(ladder[:, -1])))
    rows.sort()
    previous = np.array([r[1] for r in rows])
    final = np.array([r[2] for r in rows])
    grid_gain = final.mean() / previous.mean() - 1.0
    assert abs(grid_gain - 0.00362) < 5e-5, grid_gain
    gains = final - previous
    print("   inst      gain      share of grid gain")
    for (inst, _, _), gain, share in zip(rows, gains, gains / gains.sum()):
        marker = "  <-" if share == (gains / gains.sum()).max() else ""
        print(f"   {inst:>4}   {100 * gain / previous[inst]:>6.2f}%"
              f"   {100 * share:>8.1f}%{marker}")
    top = int(np.argmax(gains))
    print(f"\n   grid gain {100 * grid_gain:.3f}% (committed: 0.362%); "
          f"instance {rows[top][0]} carries "
          f"{100 * gains[top] / gains.sum():.1f}% of it, rising "
          f"{100 * (final[top] / previous[top] - 1):.2f}% on its own")


if __name__ == "__main__":
    self_tests()
    section_one()
    section_two()
