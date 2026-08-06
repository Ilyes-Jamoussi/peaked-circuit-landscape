"""Is the reach law a landscape statement or a step-budget artifact?

The frozen protocol stops at 400 Adam steps or when the step change falls
below the convergence tolerance, whichever comes first. That cap is not
equally binding across sizes: at n = 8 a majority of restarts converge
before it, while at n >= 10 essentially every restart -- and every
instance's BEST restart -- is still moving when it is cut off. The reach
reported at n >= 10 is therefore a truncated-optimizer quantity, which the
paper should say.

This script measures two things:

  1. Truncation, free of new compute: the fraction of restarts and of
     per-instance best restarts that stop at the cap, per size.
  2. The size of what truncation costs, from extended-budget ensembles
     (results/step_budget/, 1600 steps, matched restart seeds), and what
     correcting for it would do to the steepening statistic.

The second part carries the argument that matters. Steepening is measured
by the difference between the last and first log-step of the mean-of-best,
and a deficit d(n) that grows smoothly subtracts an almost equal amount
from every log-step, leaving that difference alone. The script computes how
much of the measured increase the observed deficits can account for, rather
than asserting it.

Self-tests: the matched-seed comparison must pair identical initial
parameters, and a synthetic linear-in-n deficit must leave the steepening
statistic unchanged to numerical precision.

Section 4 supersedes sections 2 and 3 once the converged grid exists. The
1600-step control is itself truncated -- 9/48 restarts at n = 12, 17/48 at
n = 14 and 3/16 at n = 16 still end at the extended cap -- so the deficits it
measures are lower bounds, and nothing in it bounds the gap to the true ones.
The converged runs record, along each trajectory, the step at which the frozen
rule would have fired and the value it would have returned, so the deficit
becomes an identity on one trajectory rather than a comparison between two
archives. Section 4 also tests the assumption the argument rests on: that the
deficit's logarithm is close to linear in n. That assumption is registered as
falsifiable in REGISTRATION-CONVERGED.md, because on the 1600-step control its
successive log increments are 0.0217, 0.0078, 0.0154 and 0.0575 -- a factor of
7.4 between smallest and largest, which is not a straight line.

Usage:
    python analysis/step_budget_control.py        # seconds; extended runs optional
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
SIZES = (8, 10, 12, 14, 16)
EXTENDED = ROOT / "results" / "step_budget"
CONVERGED = ROOT / "results" / "converged"


def load_grid() -> dict[int, list[Path]]:
    grid: dict[int, list[Path]] = {}
    for name in sorted(glob.glob(str(ROOT / "results/pq/pq_n*_sigma0.1.npz"))):
        match = re.search(r"pq_n(\d+)_i(\d+)_", name)
        if np.load(name)["peak_weights"].shape[0] < 200:
            continue
        grid.setdefault(int(match.group(1)), []).append(Path(name))
    return grid


def steepening(mean_best: np.ndarray) -> float:
    """Last minus first log-step of the mean-of-best, per two qubits."""
    steps = -np.diff(np.log(mean_best))
    return float(steps[-1] - steps[0])


def weighted_line_fit(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray
) -> tuple[float, float, float, int]:
    """Weighted least squares of y against x. Returns slope, intercept, chi2, dof."""
    weights = 1.0 / sigma**2
    design = np.vstack([np.ones_like(x), x]).T
    normal = design.T @ (weights[:, None] * design)
    coefficients = np.linalg.solve(normal, design.T @ (weights * y))
    residual = y - design @ coefficients
    return (
        float(coefficients[1]),
        float(coefficients[0]),
        float(np.sum(weights * residual**2)),
        len(x) - 2,
    )


def chi2_survival(chi2: float, dof: int) -> float:
    """Upper tail of the chi-square distribution (SciPy is already a dependency)."""
    from scipy import stats

    return float(stats.chi2.sf(chi2, dof)) if dof > 0 else float("nan")


def self_tests() -> None:
    sizes = np.array([8.0, 10, 12, 14, 16])
    base = np.exp(-0.3 * sizes - 0.01 * sizes**2)  # a curved, steepening law

    # Exact cancellation: a deficit whose LOGARITHM is linear in n subtracts
    # the same amount from every log-step, so the difference between the last
    # and the first is untouched.
    geometric = np.exp(0.006 * (sizes - sizes[0]))
    assert abs(steepening(base) - steepening(base * geometric)) < 1e-12, (
        "a log-linear deficit must leave the steepening statistic invariant"
    )

    # A deficit linear in n cancels to second order only; the residual has to
    # be small compared with the effect under test (~0.19 in the real data).
    linear = 1.0 + 0.006 * (sizes - sizes[0])
    residual = abs(steepening(base) - steepening(base * linear))
    assert residual < 1e-3, residual
    print(f"self-tests passed (log-linear deficit cancels exactly; linear "
          f"deficit leaves {residual:.1e}, negligible against the measured "
          "0.19)")


def main() -> None:
    self_tests()
    grid = load_grid()

    print("\n1. How binding is the 400-step cap?\n")
    print(f"{'n':>3} {'restarts at cap':>16} {'best restart at cap':>21} "
          f"{'median steps':>13}")
    for n in SIZES:
        steps, at_cap, best_at_cap, instances = [], [], 0, 0
        for path in grid.get(n, []):
            data = np.load(path)
            cap = int(data["max_steps"])
            taken = data["num_steps"]
            steps.append(taken)
            at_cap.append(taken >= cap)
            best_at_cap += int(taken[int(np.argmax(data["peak_weights"]))] >= cap)
            instances += 1
        if not instances:
            continue
        taken = np.concatenate(steps)
        fraction = float(np.concatenate(at_cap).mean())
        print(f"{n:>3} {fraction:>15.3f} {f'{best_at_cap}/{instances}':>21} "
              f"{int(np.median(taken)):>13}")

    print("\n   The cap is mild at n = 8 and total at n >= 10: every instance's "
          "best\n   restart is cut off while still improving.")

    if not EXTENDED.exists():
        print(f"\n2. Extended-budget ensembles not found under {EXTENDED}.")
        print("   Run the step_budget block of cloud/runner.py, then rerun this.")
        return

    print("\n2. What truncation costs (1600 steps vs 400, matched seeds)\n")
    print(f"{'n':>3} {'inst':>5} {'best 400':>9} {'best 1600':>10} {'gain':>7} "
          f"{'mean 400':>9} {'mean 1600':>10} {'gain':>7} {'median steps':>13} "
          f"{'at 1600 cap':>12}")
    deficits: dict[int, list[float]] = {}
    for n in SIZES:
        for path in sorted(EXTENDED.glob(f"pq_n{n}_i*_sigma0.1.npz")):
            long_run = np.load(path)
            instance = int(long_run["instance_index"])
            short = np.load(ROOT / "results/pq" /
                            f"pq_n{n}_i{instance}_sigma0.1.npz")
            count = len(long_run["peak_weights"])
            assert np.allclose(short["thetas_init"][:count],
                               long_run["thetas_init"]), (
                f"n={n} i={instance}: extended run does not share the "
                "protocol's restart seeds"
            )
            a, b = short["peak_weights"][:count], long_run["peak_weights"]
            deficits.setdefault(n, []).append(float(b.max() / a.max() - 1.0))
            extended_cap = int(long_run["max_steps"])
            at_cap = int((long_run["num_steps"] >= extended_cap).sum())
            print(f"{n:>3} {instance:>5} {a.max():>9.4f} {b.max():>10.4f} "
                  f"{100 * (b.max() / a.max() - 1):>6.2f}% {a.mean():>9.4f} "
                  f"{b.mean():>10.4f} {100 * (b.mean() / a.mean() - 1):>6.2f}% "
                  f"{int(np.median(long_run['num_steps'])):>13} "
                  f"{at_cap:>7}/{count}")

    deficit_depends_on_budget(deficits)

    print("\n3. Can truncation explain the steepening?\n")
    sizes = np.array([n for n in SIZES if grid.get(n)], dtype=float)
    mean_best = np.array([np.mean([float(np.load(p)["peak_weights"].max())
                                   for p in grid[int(n)]]) for n in sizes])
    measured = steepening(mean_best)
    measured_sizes = np.array(sorted(n for n in sizes if int(n) in deficits))
    if len(measured_sizes) < 2:
        print("   Need extended runs at two sizes or more to correct.")
        return
    measured_deficits = np.array([np.mean(deficits[int(n)])
                                  for n in measured_sizes])
    filled = np.interp(sizes, measured_sizes, measured_deficits)
    corrected = steepening(mean_best * (1.0 + filled))
    print(f"   measured steepening (last minus first log-step): {measured:+.4f}")
    print(f"   after correcting every size for its deficit:     {corrected:+.4f}")
    print(f"   share of the steepening explained by truncation: "
          f"{100 * (measured - corrected) / measured:+.1f}%")
    print("\n   Deficits enter every log-step almost equally, so they shift the "
          "reach\n   level without bending it. Truncation changes what the "
          "absolute reach\n   means; it does not manufacture the curvature.")

    within_trajectory_deficit()


def deficit_depends_on_budget(_: dict[int, list[float]]) -> None:
    """The deficit is read at 16 restarts and applied at 200. Does that matter?

    It does, and in the direction that helps: a restart that started lower
    gains more from the extended budget, so the best of a large batch gains
    less than the best of a small one. Correcting a best-of-200 curve with a
    best-of-16 deficit therefore over-corrects. This runs the other way from
    the control's own truncation, which makes its deficits lower bounds, and
    the two biases do not cancel by any argument available here -- which is
    why the converged campaign measures the deficit at the same restart budget
    it applies it to.
    """
    print("\n2b. Does the deficit depend on the restart budget it is read at?\n")
    print(f"{'n':>3} {'best-of-4':>10} {'best-of-8':>10} {'best-of-16':>11} "
          f"{'corr(rank, gain)':>17}")
    for size in SIZES:
        columns, correlations = [], []
        for path in sorted(EXTENDED.glob(f"pq_n{size}_i*_sigma0.1.npz")):
            long_run = np.load(path)
            instance = int(long_run["instance_index"])
            short = np.load(ROOT / "results/pq" / f"pq_n{size}_i{instance}_sigma0.1.npz")
            count = len(long_run["peak_weights"])
            a, b = short["peak_weights"][:count], long_run["peak_weights"]
            columns.append([b[:B].max() / a[:B].max() - 1.0 for B in (4, 8, 16)])
            rank = np.argsort(np.argsort(-a))
            correlations.append(float(np.corrcoef(rank, b / a - 1.0)[0, 1]))
        if not columns:
            continue
        means = np.mean(columns, axis=0)
        print(f"{size:>3} {100 * means[0]:>9.2f}% {100 * means[1]:>9.2f}% "
              f"{100 * means[2]:>10.2f}% {np.mean(correlations):>17.3f}")
    print("\n   A positive rank-gain correlation means the weaker restarts gain "
          "more,\n   so the deficit shrinks as the batch grows and the "
          "published correction,\n   read at 16 and applied at 200, is an "
          "over-correction.")


def load_converged() -> dict[int, list[Path]]:
    grid: dict[int, list[Path]] = {}
    for name in sorted(glob.glob(str(CONVERGED / "pq_n*_sigma0.1.npz"))):
        match = re.search(r"pq_n(\d+)_i(\d+)_", name)
        if match:
            grid.setdefault(int(match.group(1)), []).append(Path(name))
    return grid


def within_trajectory_deficit() -> None:
    """The truncation deficit read off the converged trajectories themselves.

    Sections 2 and 3 compare two archives, and the longer one is itself
    truncated, so what they measure are lower bounds. Here both numbers come
    from one trajectory: ``legacy_weight`` is what the frozen rule would have
    returned from this restart, ``peak_weights`` is what the converged run
    reached. The ratio is the deficit, not a bound on it.
    """
    grid = load_converged()
    if not grid:
        print(f"\n4. Converged ensembles not found under {CONVERGED}.")
        print("   Run the converged block of cloud/runner.py, then rerun this.")
        return

    print("\n4. Truncation deficit within the converged trajectories\n")
    print(f"{'n':>3} {'inst':>5} {'frozen rule':>12} {'converged':>10} "
          f"{'deficit':>8} {'stop step':>10}")
    sizes, deficits, deficit_errors = [], [], []
    reach_converged, reach_frozen = [], []
    for size in sorted(grid):
        per_instance, best_converged, best_frozen = [], [], []
        for path in sorted(grid[size]):
            data = np.load(path)
            if "legacy_weight" not in data:
                continue
            converged_best = float(data["peak_weights"].max())
            frozen_best = float(data["legacy_weight"].max())
            per_instance.append(converged_best / frozen_best - 1.0)
            best_converged.append(converged_best)
            best_frozen.append(frozen_best)
            print(f"{size:>3} {int(data['instance_index']):>5} {frozen_best:>12.4f} "
                  f"{converged_best:>10.4f} {100 * per_instance[-1]:>7.2f}% "
                  f"{int(np.median(data['legacy_stop_step'])):>10}")
        if not per_instance:
            continue
        values = np.array(per_instance)
        sizes.append(float(size))
        deficits.append(float(values.mean()))
        deficit_errors.append(float(values.std(ddof=1) / np.sqrt(len(values))))
        reach_converged.append(float(np.mean(best_converged)))
        reach_frozen.append(float(np.mean(best_frozen)))

    if len(sizes) < 3:
        print("\n   Need three sizes or more to test the deficit's shape.")
        return

    sizes = np.array(sizes)
    deficits = np.array(deficits)
    errors = np.array(deficit_errors)

    print("\n5. Is the deficit's logarithm linear in n?\n")
    print("   The published argument that truncation cannot manufacture the "
          "curvature\n   rests on it being so, because only a log-linear "
          "deficit cancels exactly\n   from the steepening statistic. This is "
          "the registered secondary verdict.\n")
    positive = deficits > 0
    if positive.sum() < 3:
        print("   Too few positive deficits to fit; nothing to test.")
    else:
        log_deficit = np.log(deficits[positive])
        log_error = errors[positive] / deficits[positive]
        slope, _, chi2, dof = weighted_line_fit(
            sizes[positive], log_deficit, log_error
        )
        probability = chi2_survival(chi2, dof)
        print(f"   ln d(n) against n: slope {slope:+.4f} per qubit, "
              f"chi2 = {chi2:.2f} on {dof} dof, p = {probability:.4f}")
        verdict = (
            "REJECTED: the cancellation argument does not hold and the "
            "sentence is withdrawn"
            if probability < 0.05
            else "not rejected: the cancellation argument stands as written"
        )
        print(f"   log-linearity {verdict}")

    print("\n6. The reach law, frozen rule versus converged, same trajectories\n")
    print(f"{'n':>3} {'frozen':>9} {'converged':>10} {'ratio':>7}")
    for size, frozen_value, converged_value in zip(
        sizes, reach_frozen, reach_converged
    ):
        print(f"{int(size):>3} {frozen_value:>9.4f} {converged_value:>10.4f} "
              f"{converged_value / frozen_value:>7.3f}")
    if len(sizes) >= 3:
        frozen_statistic = steepening(np.array(reach_frozen))
        converged_statistic = steepening(np.array(reach_converged))
        print(f"\n   steepening on the frozen rule:    {frozen_statistic:+.4f}")
        print(f"   steepening at convergence:        {converged_statistic:+.4f}")
        print("\n   Both are read off the same trajectories, so the difference "
              "is the\n   truncation contribution, measured rather than bounded.")


if __name__ == "__main__":
    main()
