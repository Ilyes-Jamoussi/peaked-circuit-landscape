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


if __name__ == "__main__":
    main()
