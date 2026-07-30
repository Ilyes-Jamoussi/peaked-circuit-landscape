"""Integrity gate for the P(q) data: validate before interpreting.

Checks every expected ensemble file of the first measurement pass and stops
the pipeline on any anomaly, so that no figure is ever drawn from suspect
data. Reusable as-is for future (cluster) passes.

Checks per file:
  * present, loadable, all arrays shaped consistently, no NaN/inf;
  * the announced number of restarts is complete;
  * overlap matrix symmetric, unit diagonal, values in [0, 1];
  * baseline peak weight of the right order (~2^-n, within a factor 32);
  * every restart improved on the baseline;
  * every near-optimal restart (epsilon = 0.2) peaks on the 0^n string.

Plus one global check: no FAILED marker in the run log.

Usage:
    python pq_validate.py                       # expects the full grid
    python pq_validate.py --allow-missing       # validate what is present
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

GRID_QUBIT_COUNTS = (8, 10, 12, 14)
INSTANCES = (0, 1, 2)
CONTROL = (12, 0, 1.0)  # (n, instance, init scale) of the dispersed-init run
EXPECTED_RESTARTS = 200
EPSILON = 0.2


def expected_files(directory: Path) -> list[Path]:
    files = [
        directory / f"pq_n{n}_i{i}_sigma0.1.npz"
        for n in GRID_QUBIT_COUNTS
        for i in INSTANCES
    ]
    control_n, control_i, control_scale = CONTROL
    files.append(directory / f"pq_n{control_n}_i{control_i}_sigma{control_scale:g}.npz")
    return files


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_file(path: Path) -> list[str]:
    failures: list[str] = []
    with np.load(path) as data:
        n = int(data["num_qubits"])
        restarts = len(data["peak_weights"])
        deltas = data["peak_weights"]
        overlaps = data["overlap_matrix"]
        baseline = float(data["baseline_peak_weight"])
        argmax_indices = data["argmax_indices"]

        for key in ("thetas_init", "thetas_final"):
            array = data[key]
            check(
                array.shape[0] == restarts and np.all(np.isfinite(array)),
                f"{key}: bad shape or non-finite values",
                failures,
            )
        for key in ("num_steps", "restart_seconds", "argmax_weights"):
            check(len(data[key]) == restarts, f"{key}: length mismatch", failures)

        check(
            restarts == EXPECTED_RESTARTS,
            f"expected {EXPECTED_RESTARTS} restarts, found {restarts}",
            failures,
        )
        check(np.all(np.isfinite(deltas)), "non-finite peak weights", failures)
        check(
            overlaps.shape == (restarts, restarts),
            f"overlap matrix shape {overlaps.shape}",
            failures,
        )
        check(np.allclose(overlaps, overlaps.T), "overlap matrix not symmetric", failures)
        check(
            np.allclose(np.diag(overlaps), 1.0, atol=1e-9),
            "overlap diagonal deviates from 1",
            failures,
        )
        check(
            bool(np.all(overlaps >= -1e-12) and np.all(overlaps <= 1.0 + 1e-9)),
            "overlap values outside [0, 1]",
            failures,
        )
        check(
            2.0**-n / 32 < baseline < 2.0**-n * 32,
            f"baseline {baseline:.2e} far from 2^-{n} = {2.0 ** -n:.2e}",
            failures,
        )
        check(
            bool(np.all(deltas > baseline)),
            "some restarts did not improve on the baseline",
            failures,
        )
        near_optimal = deltas >= (1.0 - EPSILON) * deltas.max()
        check(
            bool(np.all(argmax_indices[near_optimal] == 0)),
            "a near-optimal restart peaks away from the 0^n string",
            failures,
        )
    return failures


def validate_log(log_path: Path) -> list[str]:
    """No FAILED marker after the first well-formed grid-run marker.

    An aborted early launch left malformed markers ("n=8 0 instance=") and
    FAILED lines in the log; only well-formed runs are in scope.
    """
    if not log_path.exists():
        return [f"missing run log {log_path}"]
    lines = log_path.read_text().splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"=== grid run: n=\d+ instance=\d+ ===", line)
    ]
    if not starts:
        return ["run log contains no well-formed grid-run markers"]
    tail = lines[starts[0] :]
    failed = [line for line in tail if line.startswith("FAILED")]
    return [f"run log reports: {line}" for line in failed]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", type=str, default="results/pq", help="directory of pq_*.npz"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="validate present files only (grid still running)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    directory = Path(args.results)
    all_failures: list[str] = []
    present = missing = 0

    for path in expected_files(directory):
        if not path.exists():
            missing += 1
            if not args.allow_missing:
                all_failures.append(f"{path.name}: MISSING")
            continue
        present += 1
        failures = validate_file(path)
        status = "ok" if not failures else "FAIL"
        print(f"{path.name:>28}  {status}")
        all_failures.extend(f"{path.name}: {failure}" for failure in failures)

    for failure in validate_log(directory / "run.log"):
        all_failures.append(failure)

    print(f"\n{present} file(s) validated, {missing} missing")
    if all_failures:
        print("\nFAILURES:")
        for failure in all_failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
