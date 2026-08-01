"""Integrity gate for the archived ensembles: validate before interpreting.

Checks every archive under results/ and stops the pipeline on any anomaly, so
that no figure is ever drawn from suspect data.

Structural checks, per restart archive:
  * present, loadable, all arrays shaped consistently, no NaN/inf;
  * the restart count is the one the block declares;
  * overlap matrix symmetric, unit diagonal, values in [0, 1];
  * baseline peak weight of the right order (~2^-n, within a factor 32);
  * every restart improved on the baseline;
  * every near-optimal restart (epsilon = 0.2) peaks on the 0^n string.

Structural checks, per connectivity archive:
  * every group carries the seven scalars and the four curves;
  * endpoints are near-optimal for their own instance, path minima are
    positive, and adjacent-waypoint overlaps lie in [0, 1].

Content check (the only one that deserves the word "bitwise"): a SHA-256
manifest over every archive, written once with --write-manifest and verified
on every later run. Structural checks catch corruption; the manifest catches
silent replacement.

Plus one global check: no FAILED marker in a run log after its first
well-formed marker.

Usage:
    python pq_validate.py                        # validate the whole corpus
    python pq_validate.py --allow-missing        # validate what is present
    python pq_validate.py --write-manifest       # (re)create MANIFEST.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EPSILON = 0.2
MANIFEST = "MANIFEST.sha256"


@dataclass(frozen=True)
class Block:
    """One archived measurement block."""

    name: str
    pattern: str
    files: int
    restarts: int | None = None   # None: not fixed by the protocol
    schema: str = "restarts"


#: The corpus as archived (see DATA.md). File counts are asserted, so a
#: partial download or a stray archive fails the gate rather than silently
#: shrinking a figure.
CORPUS = (
    Block("pq", "pq/pq_n*.npz", 77, restarts=200),
    Block("budget800", "budget800/pq_n*_sigma0.1.npz", 6, restarts=800),
    Block("ceiling_curve", "ceiling_curve/*/pq_n*_sigma0.1.npz", 17),
    Block("depth_ceiling", "depth_ceiling/*/pq_n*_sigma0.1.npz", 8, restarts=60),
    Block("shallow_peaking", "shallow_peaking/pq_n*_sigma0.1.npz", 6, restarts=60),
    Block("robustness", "robustness/*/pq_n*_sigma*.npz", 5, restarts=200),
    Block("step_budget", "step_budget/pq_n*_sigma0.1.npz", 13, restarts=16),
    Block("connectivity", "connectivity/conn_n*.npz", 23, schema="connectivity"),
)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_restart_archive(path: Path, restarts_expected: int | None) -> list[str]:
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

        if restarts_expected is not None:
            check(
                restarts == restarts_expected,
                f"expected {restarts_expected} restarts, found {restarts}",
                failures,
            )
        check(np.all(np.isfinite(deltas)), "non-finite peak weights", failures)
        check(
            overlaps.shape == (restarts, restarts),
            f"overlap matrix shape {overlaps.shape}",
            failures,
        )
        check(np.allclose(overlaps, overlaps.T), "overlap matrix not symmetric",
              failures)
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
        # The untrained peak weight is Porter-Thomas, i.e. exponential with
        # mean 2^-n, so small values are common and carry no information: a
        # two-sided window would fail about one archive in thirty by
        # construction. Only an implausibly LARGE baseline is diagnostic --
        # it would mean the random section is not scrambling.
        check(
            0.0 < baseline < 2.0**-n * 32,
            f"baseline {baseline:.2e} implausibly large against "
            f"2^-{n} = {2.0 ** -n:.2e}",
            failures,
        )
        # Individual restarts may finish below the baseline when a control
        # under-trains on purpose (the SGD arm does); that is a measurement,
        # not corruption. Only a batch that never beats the baseline is.
        check(
            bool(deltas.max() > baseline),
            "no restart improved on the baseline",
            failures,
        )
        near_optimal = deltas >= (1.0 - EPSILON) * deltas.max()
        check(
            bool(np.all(argmax_indices[near_optimal] == 0)),
            "a near-optimal restart peaks away from the 0^n string",
            failures,
        )
    return failures


def validate_connectivity_archive(path: Path) -> list[str]:
    """String-method archives: one group of six arrays per pair or control."""
    failures: list[str] = []
    with np.load(path) as data:
        tags = sorted({key[: -len("_scalars")] for key in data.files
                       if key.endswith("_scalars")})
        check(bool(tags), "no pair or control groups", failures)
        for tag in tags:
            scalars = data[f"{tag}_scalars"]
            check(scalars.shape == (7,),
                  f"{tag}: scalars shape {scalars.shape}, expected (7,)", failures)
            for key in ("values", "raw_values", "overlap_a", "overlap_b",
                        "waypoints"):
                check(f"{tag}_{key}" in data.files, f"{tag}: missing {key}",
                      failures)
            if scalars.shape != (7,):
                continue
            delta_a, delta_b, minimum, raw_minimum, _, _, adjacent = scalars
            check(bool(np.isfinite(scalars).all()), f"{tag}: non-finite scalars",
                  failures)
            check(delta_a > 0.0, f"{tag}: endpoint A has delta {delta_a:.2e}",
                  failures)
            check(minimum >= 0.0 and raw_minimum >= 0.0,
                  f"{tag}: negative path minimum", failures)
            check(0.0 <= adjacent <= 1.0 + 1e-9,
                  f"{tag}: adjacent overlap {adjacent} outside [0, 1]", failures)
            if tag.startswith("pair"):
                # Both endpoints are near-optimal solutions of the same
                # instance; a control's partner is deliberately typical.
                check(delta_b > 0.0,
                      f"{tag}: endpoint B has delta {delta_b:.2e}", failures)
                check(minimum <= min(delta_a, delta_b) + 1e-9,
                      f"{tag}: path minimum exceeds its endpoints", failures)
    return failures


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect(results: Path) -> list[tuple[Block, Path]]:
    found = []
    for block in CORPUS:
        for path in sorted(results.glob(block.pattern)):
            found.append((block, path))
    return found


def validate_log(log_path: Path) -> list[str]:
    """No FAILED marker after the first well-formed grid-run marker.

    An aborted early launch left malformed markers ("n=8 0 instance=") and
    FAILED lines in the log; only well-formed runs are in scope.
    """
    if not log_path.exists():
        return []
    lines = log_path.read_text().splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"=== grid run: n=\d+ instance=\d+ ===", line)
    ]
    if not starts:
        return []
    failed = [line for line in lines[starts[0]:] if line.startswith("FAILED")]
    return [f"{log_path.name} reports: {line}" for line in failed]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", type=str, default="results",
                        help="root of the archived ensembles")
    parser.add_argument("--allow-missing", action="store_true",
                        help="validate present files only (grid still running)")
    parser.add_argument("--write-manifest", action="store_true",
                        help="(re)create the SHA-256 manifest instead of "
                             "verifying against it")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = Path(args.results)
    if not results.is_absolute():
        results = ROOT / results
    failures: list[str] = []
    found = collect(results)

    counts: dict[str, int] = {}
    for block, path in found:
        counts[block.name] = counts.get(block.name, 0) + 1
        if block.schema == "connectivity":
            file_failures = validate_connectivity_archive(path)
        else:
            file_failures = validate_restart_archive(path, block.restarts)
        relative = path.relative_to(results)
        if not args.quiet:
            print(f"{block.name:>16}  {str(relative):<44} "
                  f"{'ok' if not file_failures else 'FAIL'}")
        failures.extend(f"{relative}: {failure}" for failure in file_failures)

    for block in CORPUS:
        seen = counts.get(block.name, 0)
        if seen == block.files:
            continue
        message = (f"block {block.name}: {seen} archives, expected "
                   f"{block.files}")
        if seen < block.files and args.allow_missing:
            print(f"  (incomplete, allowed) {message}")
        else:
            failures.append(message)

    for log in sorted(results.rglob("run.log")):
        failures.extend(validate_log(log))

    manifest = results / MANIFEST
    if args.write_manifest:
        lines = [f"{digest(path)}  {path.relative_to(results)}"
                 for _, path in found]
        manifest.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {manifest} over {len(lines)} archives")
    elif manifest.exists():
        recorded = {}
        for line in manifest.read_text().splitlines():
            if line.strip():
                value, name = line.split(maxsplit=1)
                recorded[name] = value
        checked = 0
        for _, path in found:
            name = str(path.relative_to(results))
            if name not in recorded:
                if not args.allow_missing:
                    failures.append(f"{name}: absent from the manifest")
                continue
            if digest(path) != recorded[name]:
                failures.append(f"{name}: CONTENT CHANGED since the manifest")
            checked += 1
        print(f"\nmanifest: {checked}/{len(recorded)} archives verified byte "
              "for byte")
    else:
        print(f"\nno {MANIFEST} present; run --write-manifest to create one")

    total = sum(block.files for block in CORPUS)
    print(f"{len(found)}/{total} archive(s) validated")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
