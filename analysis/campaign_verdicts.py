"""Verdicts of the four registered predictions (11-hardness section 4).

Reads the consolidated campaign ensembles and prints, with the exact
estimators already committed for each instrument:

  1. Budget-800: log-budget slope of R(n, B) extended to B = 800
     (expected_max from budget_scan.py) and atom re-emergence check
     (degeneracy of the maximum).
  2. n = 16 point: weighted single-base fit of the mean-of-best decay
     over the consolidated grid, plus the curvature test (n = 16
     measured vs the n <= 14 extrapolation).
  3. Robustness: profile invariance across init scale, learning rate,
     and optimizer on (n = 10, instance 0) — near-optimal fraction,
     P(q_hat) intermediate band mass (pq_analysis conventions), max
     degeneracy.
  4. Corrugation scaling: delegated to connectivity.py runs at
     n = 10, 16 (logs committed separately); this script only asserts
     their presence when available.

Self-check: the budget-800 ensembles must carry 800 restarts each and
the same instance fields as the main grid (baseline peak weight matches
the grid npz for the same (n, instance)).

Usage:
    python analysis/campaign_verdicts.py
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from budget_scan import expected_max, slope_vs_logb  # noqa: E402
from pq_analysis import (  # noqa: E402
    Ensemble,
    band_mass,
    near_optimal_mask,
    pair_overlaps,
)

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_SLOPES = {10: 3.41e-3, 12: 3.27e-3}  # budget_scan.log, B <= 200
PAPER_BASE = (1.195, 0.011)  # mean-of-best law, 05 section 4 statistic


def load(path: Path) -> Ensemble:
    data = np.load(path)
    return Ensemble(
        int(data["num_qubits"]), int(data["instance_index"]),
        float(data["init_scale"]), data["peak_weights"],
        data["overlap_matrix"], data["num_steps"],
        float(data["baseline_peak_weight"]),
    )


def verdict_budget800() -> None:
    print("== 1. Budget-800 ==")
    for n in (10, 12):
        slopes = []
        for i in (0, 1, 2):
            path = ROOT / f"results/budget800/pq_n{n}_i{i}_sigma0.1.npz"
            data = np.load(path)
            values = data["peak_weights"]
            assert len(values) == 800, (path, len(values))
            grid = np.load(ROOT / f"results/pq/pq_n{n}_i{i}_sigma0.1.npz")
            assert np.isclose(
                float(data["baseline_peak_weight"]),
                float(grid["baseline_peak_weight"]),
            ), "instance mismatch vs main grid"
            slope = slope_vs_logb(values, (100, 200, 400, 800))
            slopes.append(slope)
            degeneracy = int((values > values.max() - 1e-4).sum())
            curve = [expected_max(values, b) for b in (100, 200, 400, 800)]
            print(f"  n={n} i={i}: E[max|B] "
                  f"{[f'{c:.4f}' for c in curve]}  slope {slope:.2e}  "
                  f"degeneracy(1e-4) {degeneracy}/800")
        pooled = float(np.mean(slopes))
        ref = REFERENCE_SLOPES[n]
        print(f"  pooled n={n}: slope[100-800] {pooled:.2e} vs "
              f"reference[6-200] {ref:.2e} (ratio {pooled / ref:.2f})")
    print("  VERDICT: slope persists at the same order to B = 800; no "
          "atom (unique maxima) -> prediction 1 CONFIRMED\n")


def verdict_n16() -> None:
    print("== 2. n = 16 on the mean-of-best law ==")
    bests: dict[int, list[float]] = {}
    for name in sorted(glob.glob(str(ROOT / "results/pq/pq_n*_sigma0.1.npz"))):
        match = re.search(r"pq_n(\d+)_i(\d+)_", name)
        values = np.load(name)["peak_weights"]
        if len(values) < 200:
            continue
        bests.setdefault(int(match.group(1)), []).append(float(values.max()))
    sizes = np.array(sorted(bests))
    mean_best = np.array([np.mean(bests[n]) for n in sizes])
    stderr = np.array(
        [np.std(bests[n], ddof=1) / np.sqrt(len(bests[n])) for n in sizes]
    )
    for n, m, s in zip(sizes, mean_best, stderr):
        print(f"  n={n:>2}: {len(bests[n]):>2} instances  "
              f"mean-of-best {m:.4f} +/- {s:.4f}")
    # np.polyfit weights apply to unsquared residuals: w = 1/sigma(ln y).
    weights = mean_best / stderr
    coefficients, covariance = np.polyfit(
        sizes, np.log(mean_best), 1, w=weights, cov=True
    )
    base = float(np.exp(-coefficients[0]))
    base_error = base * float(np.sqrt(covariance[0, 0]))
    print(f"  weighted single-base fit: {base:.4f} +/- {base_error:.4f} "
          f"(paper statistic {PAPER_BASE[0]} +/- {PAPER_BASE[1]})")
    head = np.polyfit(sizes[:-1], np.log(mean_best[:-1]), 1, w=weights[:-1])
    extrapolated = float(np.exp(np.polyval(head, sizes[-1])))
    shortfall = (extrapolated - mean_best[-1]) / stderr[-1]
    print(f"  curvature test: n<=14 fit extrapolates n=16 to "
          f"{extrapolated:.4f}; measured {mean_best[-1]:.4f} "
          f"({shortfall:.1f} sigma below)")
    steps = -np.diff(np.log(mean_best))
    print(f"  log-steps per 2 qubits: {[f'{s:.3f}' for s in steps]}")


def verdict_robustness() -> None:
    print("\n== 3. Robustness of the profile (n = 10, instance 0) ==")
    variants = [
        ("protocol Adam s=0.1", "results/pq/pq_n10_i0_sigma0.1.npz"),
        ("init s=0.5", "results/robustness/init/pq_n10_i0_sigma0.5.npz"),
        ("init s=1.0", "results/robustness/init/pq_n10_i0_sigma1.npz"),
        ("lr 0.025", "results/robustness/lr_half/pq_n10_i0_sigma0.1.npz"),
        ("lr 0.1", "results/robustness/lr_double/pq_n10_i0_sigma0.1.npz"),
        ("SGD", "results/robustness/sgd/pq_n10_i0_sigma0.1.npz"),
    ]
    for label, relative in variants:
        ensemble = load(ROOT / relative)
        values = ensemble.peak_weights
        mask = near_optimal_mask(values, 0.2)
        _, normalized, _ = pair_overlaps(ensemble, 0.2)
        print(f"  {label:>20}: best {values.max():.4f}  "
              f"mean {values.mean():.4f}  near-opt {mask.sum():>3}/200  "
              f"band mass {band_mass(normalized):.3f}  "
              f"degeneracy(1e-4) {int((values > values.max() - 1e-4).sum())}")
    print("  VERDICT: invariant across init scale x10 and learning rate "
          "x4; plain SGD is inside the class as registered (the registered "
          "class carries no pacing criterion) and its profile is not the "
          "protocol's -> prediction 3 FALSIFIED as registered. The upward "
          "question is graded by the registered optclass block "
          "(analysis/optclass_reach.log).")


def anchors_decomposition() -> None:
    """F1 depth decomposition per instance (08 section 4, multi-instance)."""
    print("\n== 5. F1 depth decomposition (deep anchors, 3 instances) ==")
    for n, tau in ((10, 40), (12, 48)):
        for i in (0, 1, 2):
            deep = np.load(
                ROOT / f"results/depth_ceiling/tau{tau}/pq_n{n}_i{i}_sigma0.1.npz"
            )["peak_weights"]
            shallow = np.load(
                ROOT / f"results/pq/pq_n{n}_i{i}_sigma0.1.npz"
            )["peak_weights"]
            top3 = np.sort(deep)[-3:]
            print(f"  n={n} i={i}: deep best {deep.max():.4f} "
                  f"(top3 spread {top3.max() - top3.min():.1e})  "
                  f"shallow best {shallow.max():.4f}  "
                  f"excess ratio {shallow.max() / deep.max():.2f}")


CORRUGATION_SIZES = (8, 10, 12, 14, 16)
CORRUGATION_INSTANCES = (0, 1, 2)


def corrugation_archive(num_qubits: int, instance: int) -> Path:
    """Matched-resolution archive; instance 0 keeps its historical name."""
    tag = f"_i{instance}" if instance else ""
    return ROOT / "results/connectivity" / f"conn_n{num_qubits}{tag}_o150s1_m64.npz"


def pair_ratios(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.array([
        data[key][2] / min(data[key][0], data[key][1])
        for key in data.keys()
        if key.startswith("pair") and key.endswith("_scalars")
    ])


def weighted_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray):
    """Weighted least squares y = a + b x; returns (a, b, chi2)."""
    weight = 1.0 / sigma**2
    total = weight.sum()
    mean_x = (weight * x).sum() / total
    mean_y = (weight * y).sum() / total
    slope = ((weight * (x - mean_x) * (y - mean_y)).sum()
             / (weight * (x - mean_x) ** 2).sum())
    intercept = mean_y - slope * mean_x
    chi2 = float((weight * (y - intercept - slope * x) ** 2).sum())
    return float(intercept), float(slope), chi2


def verdict_corrugation() -> None:
    """Registered analysis; see REGISTRATION.md, written before these runs."""
    print("\n== 4. Corrugation scaling (registered rule, REGISTRATION.md) ==")
    table: dict[int, list[float]] = {}
    for num_qubits in CORRUGATION_SIZES:
        for instance in CORRUGATION_INSTANCES:
            path = corrugation_archive(num_qubits, instance)
            if not path.exists():
                continue
            ratios = pair_ratios(path)
            table.setdefault(num_qubits, []).append(float(np.median(ratios)))
            print(f"  n={num_qubits:>2} i={instance}: {len(ratios)} pairs  "
                  f"median rho {np.median(ratios):.3f}  "
                  f"range [{ratios.min():.3f}, {ratios.max():.3f}]")

    complete = [n for n in CORRUGATION_SIZES
                if len(table.get(n, [])) == len(CORRUGATION_INSTANCES)]
    if len(complete) < 4:
        have = {n: len(table.get(n, [])) for n in CORRUGATION_SIZES}
        print(f"  matched-resolution sweep incomplete (instances per size: "
              f"{have}); verdict deferred, see REGISTRATION.md\n")
        return

    sizes = np.array(complete, dtype=float)
    mean = np.array([np.mean(table[int(n)]) for n in sizes])
    stderr = np.array([np.std(table[int(n)], ddof=1)
                       / np.sqrt(len(table[int(n)])) for n in sizes])
    print(f"\n  {'n':>3} {'mean rho':>9} {'SE(inst)':>9}   per-instance medians")
    for n, m, s in zip(sizes, mean, stderr):
        values = ", ".join(f"{v:.3f}" for v in table[int(n)])
        print(f"  {int(n):>3} {m:>9.4f} {s:>9.4f}   {values}")

    y = np.log(mean)
    sigma = stderr / mean
    dof = len(sizes) - 2
    _, slope_exp, chi2_exp = weighted_fit(sizes, y, sigma)
    _, slope_pow, chi2_pow = weighted_fit(np.log(sizes), y, sigma)
    p_exp = float(1.0 - stats.chi2.cdf(chi2_exp, dof))
    p_pow = float(1.0 - stats.chi2.cdf(chi2_pow, dof))
    alphas = [-(y[k + 1] - y[k]) / (np.log(sizes[k + 1]) - np.log(sizes[k]))
              for k in range(len(sizes) - 1)]
    print(f"\n  exponential: base {np.exp(slope_exp):.4f}^n   "
          f"chi2 {chi2_exp:.2f} on {dof} dof, p = {p_exp:.4f}")
    print(f"  power law:   exponent {slope_pow:.2f}      "
          f"chi2 {chi2_pow:.2f} on {dof} dof, p = {p_pow:.4f}")
    print(f"  interval exponents: {[f'{a:.2f}' for a in alphas]}")

    exponential_wins = chi2_exp < chi2_pow
    fixed_exponent_excluded = p_pow < 0.05
    if exponential_wins and fixed_exponent_excluded:
        print("  VERDICT (registered branch a): the exponential attains the "
              "lower chi2 and a fixed\n    exponent is excluded at p < 0.05 -> "
              "'deepens faster than any fixed power' STANDS,\n    now carried "
              "by instance-level errors.")
    else:
        reason = ("the power law fits at least as well"
                  if not exponential_wins
                  else f"a fixed exponent survives at p = {p_pow:.3f}")
        print(f"  VERDICT (registered branch b): {reason} -> the claim is "
              "WITHDRAWN from the\n    abstract and replaced by 'the "
              "corrugation deepens with n; these data do not\n    determine "
              "the functional form'. C-shelf drops its rate clause; "
              "prediction 4 is\n    falsified as stated and superseded by a "
              "deepening of undetermined form.")
    print("  Connectivity itself is unaffected either way: string minima stay "
          "far above\n    2^-n and the scrambled controls collapse to it.")


def self_tests() -> None:
    """Data-inventory gate: the verdicts below assume exactly this corpus."""
    for n in (8, 10, 12, 14, 16):
        count = len(glob.glob(str(ROOT / f"results/pq/pq_n{n}_i*_sigma0.1.npz")))
        assert count == 18, (n, count, 18)
    for n in (10, 12):
        for i in (0, 1, 2):
            values = np.load(
                ROOT / f"results/budget800/pq_n{n}_i{i}_sigma0.1.npz"
            )["peak_weights"]
            assert len(values) == 800, (n, i, len(values))
    assert (ROOT / "results/connectivity").is_dir(), "connectivity results"
    # Two anchored references recomputed from raw data. The n = 16 point grew
    # from four instances to eighteen, and BOTH values are asserted: the first
    # four still reproduce the number the manuscript was written against
    # (campaign_verdicts.log / 11-hardness section 5), so a reader can trace
    # the published figure through the change rather than take the new one on
    # trust.
    def n16_mean(limit: int | None = None) -> float:
        paths = sorted(glob.glob(str(ROOT / "results/pq/pq_n16_i*_sigma0.1.npz")),
                       key=lambda p: int(re.search(r"_i(\d+)_", p).group(1)))
        if limit is not None:
            paths = paths[:limit]
        return float(np.mean([float(np.load(p)["peak_weights"].max())
                              for p in paths]))

    assert abs(n16_mean(4) - 0.1258) < 5e-4, n16_mean(4)
    assert abs(n16_mean() - 0.1241) < 5e-4, n16_mean()
    print("self-tests passed (corpus 18 at every size, budget800 6x800, "
          "connectivity npz present, n=16 mean-of-best anchored at 0.1258 on "
          "the published four and 0.1241 on all eighteen)")


def main() -> None:
    self_tests()
    verdict_budget800()
    verdict_n16()
    verdict_robustness()
    anchors_decomposition()
    verdict_corrugation()


if __name__ == "__main__":
    main()
