"""Is the reach law still a fixed-base law once the optimizer has converged?

The published reach series is measured under the frozen protocol, which cuts
every restart at 400 Adam steps. That cap binds 94-99 % of restarts and every
instance's best restart at n >= 10, so each published level is a lower bound
whose shortfall grows with n -- and a shortfall that grows with n has the same
signature as a base that grows with n. The steepening cannot be read as a
statement about the landscape until reach is measured at convergence.

This script reports the reach law of the converged campaign (results/converged:
ladder to 12800 steps, learning-rate floor, relative-gain stop, final polish,
B = 32 restarts, 18 instances per size including n = 16) next to the frozen
grid (results/pq), statistic for statistic, so the two read off one page.

The primary statistic is R(n) = mean over instances of E[max of B0 = 16
restarts], the exact without-replacement estimator of budget_scan. The error
is taken over INSTANCES: instances are the independent draws from the circuit
ensemble, restarts are not. The secondary statistic is the archive's own best,
which is what the manuscript publishes, kept so the two grids can also be
compared at the level the paper quotes.

What the script decides: a weighted fit of ln R against n on n <= 14, with its
chi2 on 2 degrees of freedom. On the frozen grid that fit rejects the
fixed-base law at p = 0.009. Whether the converged grid still rejects it is
the question the campaign exists to answer, and nothing here presumes the
answer: both grids are run through identical code and printed side by side.

Self-tests: a synthetic fixed-base law is not rejected, a curved law is, and
the budget estimator returns the constant on a constant ensemble.

Usage:
    python analysis/converged_reach.py     # seconds; converged grid optional
"""

from __future__ import annotations

import glob
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from budget_scan import expected_max  # noqa: E402

FROZEN = ROOT / "results" / "pq"
CONVERGED = ROOT / "results" / "converged"
FROZEN_RESTARTS = 200
CONVERGED_RESTARTS = 32
BASE_BUDGET = 16  # primary budget, reachable on both grids
FIT_MAX_SIZE = 14  # the manuscript tests the fixed base on n <= 14

#: What pq_experiment._collect_converged writes for every converged restart,
#: and therefore what distinguishes a converged archive from a frozen one. The
#: two protocols write the same file name into different directories and share
#: every key this script reads, so the directory is not evidence of anything:
#: without this gate a frozen archive dropped into results/converged would be
#: fitted, tested and printed as converged data, and the verdict of the whole
#: campaign would rest on 400-step runs. analysis/convergence_diagnostics.py
#: gates the converged grid on the same tuple.
INSTRUMENTATION_KEYS = (
    "ladder_weights",
    "running_max",
    "stop_reasons",
    "legacy_weight",
    "legacy_stop_step",
)

RUN_CONVERGED = (
    "python pq_experiment.py --converged --restarts 32 --instance I "
    "--num-qubits N --output results/converged --restart-cache CACHE"
)


def load_grid(directory: Path, min_restarts: int,
              require_instrumentation: bool = False
              ) -> dict[int, list[np.ndarray]]:
    """Peak-weight ensembles by size, skipping under-sized batches.

    ``require_instrumentation`` is set for the converged grid and refuses any
    archive that does not carry the converged readout of INSTRUMENTATION_KEYS.
    An under-sized batch is skipped, because a half-finished size is a normal
    state of a running campaign; a frozen archive in the converged directory is
    not, so it stops the script by name instead. ``peak_weights`` alone cannot
    tell the two protocols apart, and reading a 400-step archive as converged
    would answer the campaign's question with the data it was built to replace.
    """
    grid: dict[int, list[np.ndarray]] = {}
    for name in sorted(glob.glob(str(directory / "pq_n*_sigma0.1.npz"))):
        blob = np.load(name)
        if require_instrumentation:
            missing = [key for key in INSTRUMENTATION_KEYS
                       if key not in blob.files]
            if missing:
                raise ValueError(
                    f"{Path(name).name}: read from the converged grid "
                    f"({directory}) but the converged instrumentation key "
                    f"'{missing[0]}' is absent"
                    + (f" (and {len(missing) - 1} more of "
                       f"{list(INSTRUMENTATION_KEYS)})" if len(missing) > 1
                       else "")
                    + ". A frozen-protocol archive cannot be reported as "
                    "converged data; move it out of that directory."
                )
        values = blob["peak_weights"]
        if len(values) < min_restarts:
            continue
        num_qubits = int(re.search(r"pq_n(\d+)_i(\d+)_", name).group(1))
        grid.setdefault(num_qubits, []).append(values)
    return grid


def reach(grid: dict[int, list[np.ndarray]], budget: int | None):
    """Reach per size: mean over instances, and the error over instances.

    ``budget`` selects the exact E[max of B] estimator; ``None`` selects the
    archive's raw best, the statistic the manuscript publishes.
    """
    sizes = np.array(sorted(grid), dtype=float)
    per_instance = [
        np.array([
            float(v.max()) if budget is None else expected_max(v, budget)
            for v in grid[int(n)]
        ])
        for n in sizes
    ]
    mean = np.array([values.mean() for values in per_instance])
    sem = np.array([values.std(ddof=1) / np.sqrt(len(values))
                    for values in per_instance])
    counts = np.array([len(values) for values in per_instance])
    return sizes, mean, sem, counts


def weighted_fit(x: np.ndarray, y: np.ndarray, sy: np.ndarray):
    """Weighted least squares of y on x; returns fit, slope error and chi2."""
    w = 1.0 / sy**2
    s, sx, sxx = w.sum(), (w * x).sum(), (w * x * x).sum()
    sy_, sxy = (w * y).sum(), (w * x * y).sum()
    det = s * sxx - sx**2
    slope = (s * sxy - sx * sy_) / det
    intercept = (sxx * sy_ - sx * sxy) / det
    residual = y - (intercept + slope * x)
    return intercept, slope, float(np.sqrt(s / det)), float((w * residual**2).sum())


def fixed_base_test(sizes: np.ndarray, mean: np.ndarray, sem: np.ndarray) -> dict:
    """Fit ln R = const - n ln(base) on n <= 14 and score the misfit."""
    head = sizes <= FIT_MAX_SIZE
    x, y, sy = sizes[head], np.log(mean[head]), (sem / mean)[head]
    intercept, slope, slope_err, chi2 = weighted_fit(x, y, sy)
    dof = len(x) - 2
    return {
        "base": float(np.exp(-slope)),
        "base_err": float(np.exp(-slope) * slope_err),
        "chi2": chi2,
        "dof": dof,
        "p": float(1 - stats.chi2.cdf(chi2, dof)) if dof > 0 else float("nan"),
        "intercept": intercept,
        "slope": slope,
    }


def steepening(mean: np.ndarray, sem: np.ndarray, head: int | None = None):
    """Last minus first log-step of the reach series, and its error.

    ``head`` truncates the series to its first ``head`` sizes before the
    statistic is formed, and it is not cosmetic. The registered statistic of
    REGISTRATION-CONVERGED.md is ``log-step(12 -> 14) - log-step(8 -> 10)``,
    that is ``head = 4``, over the same head of the grid the chi-square test
    uses; the whole series instead gives ``log-step(14 -> 16) -
    log-step(8 -> 10)``. On the published grid the two read 0.191 and 0.241, a
    24% difference, and branches (a) and (c) of the verdict turn on T at two
    standard errors. The window is therefore printed wherever T is.

    The statistic is a fixed linear combination of the log levels, so its
    error follows from the per-size errors even when a level enters twice,
    which it does whenever fewer than four sizes are available. Two sizes
    leave a single log-step and one size leaves none, so there the statistic
    is reported as absent: an exact +0.0000 would read as "no steepening"
    from a grid that cannot measure it.
    """
    if head is not None:
        mean, sem = mean[:head], sem[:head]
    if len(mean) < 3:
        return None, None
    y, sy = np.log(mean), sem / mean
    coefficients = np.zeros(len(y))
    coefficients[-2] += 1.0
    coefficients[-1] -= 1.0
    coefficients[0] -= 1.0
    coefficients[1] += 1.0
    steps = -np.diff(y)
    return float(steps[-1] - steps[0]), float(
        np.sqrt(np.sum((coefficients * sy) ** 2))
    )


def extrapolate_last(sizes: np.ndarray, mean: np.ndarray, sem: np.ndarray) -> dict:
    """Where the n <= 14 fit puts the largest size, and how far below it lands."""
    fit = fixed_base_test(sizes, mean, sem)
    largest = sizes[-1]
    if largest <= FIT_MAX_SIZE:
        return {}
    predicted = fit["intercept"] + fit["slope"] * largest
    measured, error = np.log(mean[-1]), (sem / mean)[-1]
    return {
        "size": largest,
        "predicted": float(np.exp(predicted)),
        "measured": float(mean[-1]),
        "shortfall": float(np.exp(measured - predicted) - 1.0),
        "sigma": float((predicted - measured) / error),
    }


def cell(value, spec: str, width: int) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}{spec}}"


def self_tests() -> None:
    constant = np.full(64, 0.375)
    assert abs(expected_max(constant, BASE_BUDGET) - 0.375) < 1e-12, (
        "E[max of B] on a constant ensemble must return that constant"
    )

    sizes = np.array([8.0, 10, 12, 14, 16])
    rng = np.random.default_rng(17)
    relative_error = 0.015
    fixed = 5e-4 * 1.19 ** (34 - sizes)

    # Size of the test, measured rather than assumed: a fixed-base law
    # perturbed at the level of its own error bars must be rejected at close
    # to the nominal rate, or the verdict on the converged grid means nothing.
    rejected = 0
    for _ in range(400):
        noisy = fixed * (1.0 + relative_error * rng.standard_normal(len(sizes)))
        if fixed_base_test(sizes, noisy, relative_error * noisy)["p"] < 0.05:
            rejected += 1
    rate = rejected / 400
    assert rate < 0.12, f"fixed-base law rejected {rate:.3f} of the time"

    # A law whose base grows with n must be rejected, at the same error bars.
    curved = np.exp(-0.3 * sizes - 0.01 * sizes**2)
    curved_result = fixed_base_test(sizes, curved, relative_error * curved)
    assert curved_result["p"] < 0.01, curved_result

    # The steepening statistic is zero on a fixed-base law by construction,
    # and absent on a partial grid: two sizes hold one log-step, one size
    # holds none, and an exact zero there would claim a fixed base.
    assert abs(steepening(fixed, relative_error * fixed)[0]) < 1e-12
    for cut in (1, 2):
        assert steepening(fixed[:cut], (relative_error * fixed)[:cut]) == (
            None, None
        ), f"steepening must be absent on {cut} size(s)"

    # A frozen archive in the converged directory must be refused by name, and
    # not read: the two protocols write identical file names and share every
    # key this script reads, so nothing but the instrumentation tells them
    # apart. Both directions are checked, since a gate that refuses everything
    # would pass a one-sided test and leave the campaign unreadable.
    frozen_keys = {"peak_weights": np.full(CONVERGED_RESTARTS, 0.3),
                   "thetas_init": np.zeros((CONVERGED_RESTARTS, 2)),
                   "instance_index": 0}
    converged_keys = dict(
        frozen_keys,
        ladder_weights=np.full((CONVERGED_RESTARTS, 6), 0.3),
        running_max=np.full(CONVERGED_RESTARTS, 0.3),
        stop_reasons=np.array(["rel_tol"] * CONVERGED_RESTARTS, dtype="<U12"),
        legacy_weight=np.full(CONVERGED_RESTARTS, 0.28),
        legacy_stop_step=np.full(CONVERGED_RESTARTS, 400, dtype=np.int64),
    )
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        name = "pq_n8_i0_sigma0.1.npz"
        np.savez_compressed(directory / name, **frozen_keys)
        try:
            load_grid(directory, CONVERGED_RESTARTS,
                      require_instrumentation=True)
        except ValueError as refusal:
            message = str(refusal)
        else:
            raise AssertionError("a frozen archive was read as converged data")
        assert name in message, message
        assert "ladder_weights" in message, message

        # The same file is still legitimate frozen data, so the frozen grid
        # must keep reading it: the gate is on the caller, not on the file.
        assert len(load_grid(directory, CONVERGED_RESTARTS)[8]) == 1

        np.savez_compressed(directory / name, **converged_keys)
        accepted = load_grid(directory, CONVERGED_RESTARTS,
                             require_instrumentation=True)
        assert len(accepted[8]) == 1, accepted

        # A missing key anywhere in the tuple is caught, not just the first.
        for key in INSTRUMENTATION_KEYS:
            partial = {k: v for k, v in converged_keys.items() if k != key}
            np.savez_compressed(directory / name, **partial)
            try:
                load_grid(directory, CONVERGED_RESTARTS,
                          require_instrumentation=True)
            except ValueError as refusal:
                assert key in str(refusal), (key, str(refusal))
            else:
                raise AssertionError(f"a missing '{key}' was not refused")

    print(
        f"self-tests passed (fixed base rejected {rate:.3f} of 400 draws at a "
        f"nominal 0.05; curved law p = {curved_result['p']:.1e}, rejected; "
        "T = 0 on a fixed base and absent below three sizes; E[max of B] "
        "exact on a constant ensemble;\na frozen archive in the converged "
        f"grid is refused by name and each of the {len(INSTRUMENTATION_KEYS)} "
        "instrumentation keys is required)"
    )


def main() -> None:
    self_tests()

    grids = {"frozen": load_grid(FROZEN, FROZEN_RESTARTS)}
    if CONVERGED.exists():
        grids["converged"] = load_grid(CONVERGED, CONVERGED_RESTARTS,
                                       require_instrumentation=True)
    if not grids["frozen"]:
        print(f"\nNo frozen grid under {FROZEN}; nothing to compare.")
        return
    if not grids.get("converged"):
        print(f"\nConverged grid not found under {CONVERGED}.")
        print(f"   Run, per size and instance:\n     {RUN_CONVERGED}")
        print("   18 instances at each of n = 8, 10, 12, 14, 16. Until then "
              "only the\n   frozen grid is reported, and every converged "
              "column reads '-'.")
        grids.pop("converged", None)

    statistics = {}
    for label, grid in grids.items():
        statistics[label] = {
            "primary": reach(grid, BASE_BUDGET),
            "archive": reach(grid, None),
        }

    print("\n1. Reach level, frozen grid vs converged grid\n")
    print(f"   R16 = mean over instances of E[max of B = {BASE_BUDGET} "
          "restarts]; the error is\n   the standard error over instances. "
          "'archive' is the raw best over all\n   stored restarts, the "
          f"statistic the manuscript publishes -- the two grids\n   hold "
          f"{FROZEN_RESTARTS} and {CONVERGED_RESTARTS} restarts, so only the "
          "R16 columns compare like with like.\n")
    print((f"{'':>4} {'frozen (results/pq)':^33}   "
           f"{'converged (results/converged)':^33}").rstrip())
    print(f"{'n':>4} {'inst':>6} {'R16':>8} {'+/-':>8} {'archive':>7}   "
          f"{'inst':>6} {'R16':>8} {'+/-':>8} {'archive':>7}")
    all_sizes = sorted({int(n) for label in statistics
                        for n in statistics[label]["primary"][0]})
    for num_qubits in all_sizes:
        row = f"{num_qubits:>4}"
        for label in ("frozen", "converged"):
            found = statistics.get(label)
            index = None
            if found is not None:
                sizes = found["primary"][0]
                match = np.flatnonzero(sizes == num_qubits)
                index = int(match[0]) if len(match) else None
            if index is None:
                row += (f" {cell(None, '', 6)} {cell(None, '', 8)}"
                        f" {cell(None, '', 8)} {cell(None, '', 7)}  ")
                continue
            _, mean, sem, counts = found["primary"]
            archive = found["archive"][1]
            row += (f" {counts[index]:>6} {mean[index]:>8.4f}"
                    f" {sem[index]:>8.4f} {archive[index]:>7.4f}  ")
        print(row.rstrip())

    print("\n2. Local base per interval (exp of half the log-step)\n")
    print((f"{'':>11} {'frozen':^19}   {'converged':^19}").rstrip())
    print(f"{'interval':>11} {'R16':>9} {'archive':>9}   "
          f"{'R16':>9} {'archive':>9}")
    intervals = [(all_sizes[i], all_sizes[i + 1])
                 for i in range(len(all_sizes) - 1)]
    for low, high in intervals:
        row = f"{f'{low} -> {high}':>11}"
        for label in ("frozen", "converged"):
            found = statistics.get(label)
            for key in ("primary", "archive"):
                value = None
                if found is not None:
                    sizes, mean = found[key][0], found[key][1]
                    lo = np.flatnonzero(sizes == low)
                    hi = np.flatnonzero(sizes == high)
                    if len(lo) and len(hi):
                        step = np.log(mean[lo[0]]) - np.log(mean[hi[0]])
                        value = float(np.exp(step / (high - low)))
                row += f" {cell(value, '.3f', 9)}"
            row += "  "
        print(row.rstrip())

    print(f"\n3. Fixed-base law on n <= {FIT_MAX_SIZE}, and the steepening\n")
    print(f"{'grid':>10} {'stat':>8} {'base':>7} {'+/-':>7} {'chi2':>7} "
          f"{'dof':>4} {'p':>8} {'T(reg)':>9} {'+/-':>7} {'T(full)':>9} "
          f"{'+/-':>7}")
    for label in ("frozen", "converged"):
        found = statistics.get(label)
        for key, name in (("primary", "R16"), ("archive", "archive")):
            if found is None:
                print(f"{label:>10} {name:>8} {cell(None, '', 7)} "
                      f"{cell(None, '', 7)} {cell(None, '', 7)} "
                      f"{cell(None, '', 4)} {cell(None, '', 8)} "
                      f"{cell(None, '', 9)} {cell(None, '', 7)} "
                      f"{cell(None, '', 9)} {cell(None, '', 7)}")
                continue
            sizes, mean, sem, _ = found[key]
            test = fixed_base_test(sizes, mean, sem)
            head = int(np.count_nonzero(sizes <= FIT_MAX_SIZE))
            registered, registered_error = steepening(mean, sem, head=head)
            full, full_error = steepening(mean, sem)
            signed = ("-" if registered is None else f"{registered:+.4f}")
            signed_full = "-" if full is None else f"{full:+.4f}"
            print(f"{label:>10} {name:>8} {test['base']:>7.3f} "
                  f"{test['base_err']:>7.3f} {test['chi2']:>7.2f} "
                  f"{test['dof']:>4} {test['p']:>8.4f} "
                  f"{signed:>9} {cell(registered_error, '.4f', 7)} "
                  f"{signed_full:>9} {cell(full_error, '.4f', 7)}")
    print(f"\n   T(reg) is the registered statistic, log-step(12 -> 14) minus\n"
          f"   log-step(8 -> 10): the same head of the grid, n <= "
          f"{FIT_MAX_SIZE}, that the\n   chi-square column tests, and the one "
          "branches (a) and (c) of\n   REGISTRATION-CONVERGED.md are decided "
          "on. T(full) carries the largest\n   size as well and is reported "
          "only. Both are zero on a fixed base.")

    print(f"\n4. The largest size against the extrapolation from n <= "
          f"{FIT_MAX_SIZE}\n")
    print(f"{'grid':>10} {'stat':>8} {'n':>4} {'extrapolated':>13} "
          f"{'measured':>9} {'shortfall':>10} {'sigma':>7}")
    for label in ("frozen", "converged"):
        found = statistics.get(label)
        for key, name in (("primary", "R16"), ("archive", "archive")):
            gap = {} if found is None else extrapolate_last(*found[key][:3])
            if not gap:
                print(f"{label:>10} {name:>8} {cell(None, '', 4)} "
                      f"{cell(None, '', 13)} {cell(None, '', 9)} "
                      f"{cell(None, '', 10)} {cell(None, '', 7)}")
                continue
            print(f"{label:>10} {name:>8} {int(gap['size']):>4} "
                  f"{gap['predicted']:>13.4f} {gap['measured']:>9.4f} "
                  f"{100 * gap['shortfall']:>9.1f}% {gap['sigma']:>7.1f}")

    print("\n   The frozen grid's shortfall at the largest size is the "
          "manuscript's\n   headline. It survives only if it also appears on "
          "the converged grid,\n   where the 400-step cap can no longer "
          "manufacture a size-dependent\n   deficit. Read rows 3 and 4 of the "
          "converged block, not the frozen ones.")


if __name__ == "__main__":
    main()
