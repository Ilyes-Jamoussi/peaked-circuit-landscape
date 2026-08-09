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

Structural checks, per converged-protocol archive (those carrying the ladder
readout), on top of the restart checks:
  * the settings keys and the readout keys are each all-or-nothing, and a
    readout never appears without the settings that produced it;
  * every stop reason is in the vocabulary of the archive's own optimizer:
    "cap", "rel_tol", "legacy_tol" for the instrumented Adam loop, the scipy
    exit status for the L-BFGS arm, which has no step schedule to stop on;
  * the ladder is strictly increasing and ends at or below max_steps;
  * ladder_weights is a running maximum, so it never decreases along the
    ladder, and its NaN padding is a suffix (the restart had already stopped);
  * legacy_weight <= running_max <= peak_weight, and polish_gain >= 0: the
    within-trajectory readout can never exceed the run that contains it, and
    polishing can never lose ground;
  * the frozen stopping rule fired at a step the trajectory actually reached,
    and no later than the frozen cap -- or, on the L-BFGS arm where that rule
    has no schedule to read, that both its sentinels are exactly the
    documented (-1, NaN) pair and never a plausible-looking step and weight.

Content check: a SHA-256 manifest over every archive, written once with
--write-manifest and verified on every later run. It pins the archived bytes,
so it catches the silent replacement that structural checks cannot see.

What the manifest is *not* is a reproducibility contract. Two of the arrays,
restart_seconds and elapsed_seconds, are wall-clock measurements, so no
archive has ever been reproducible byte for byte -- not on another machine,
not from the same seed on the same machine. The contract that does hold, and
that the campaign depends on, is that every *other* array is bit-identical
across machines of the same architecture. --compare-arrays is what checks it:
point it at a second copy of the corpus and it reports, file by file, which
arrays moved and by how much. Run it whenever a block changes machine.

Plus one global check: no FAILED marker in a run log after its first
well-formed marker.

Usage:
    python pq_validate.py                        # validate the whole corpus
    python pq_validate.py --allow-missing        # validate what is present
    python pq_validate.py --write-manifest       # (re)create MANIFEST.sha256
    python pq_validate.py --compare-arrays DIR   # binary portability against DIR
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

#: Two groups of keys are appended past the frozen ones, and they are not
#: appended together. The settings group goes in whenever the run is not the
#: frozen protocol -- an L-BFGS or Haar-init arm carries it. The readout group
#: goes in only when the ladder is on, so an arm can legitimately carry the
#: first without the second. Each group is all-or-nothing.
CONVERGED_SETTINGS_KEYS = (
    "min_learning_rate",
    "ladder",
    "rel_tol",
    "polish_steps",
    "init_mode",
    "optimizer",
)
CONVERGED_READOUT_KEYS = (
    "stop_reasons",
    "ladder_weights",
    "running_max",
    "polish_gain",
    "legacy_stop_step",
    "legacy_weight",
)
#: Two stop-reason vocabularies, one per optimizer class. The instrumented
#: Adam loop leaves for one of three reasons; L-BFGS-B has no step schedule to
#: stop on, so lbfgs_restart.py reports the scipy exit status instead. An
#: archive is held to the vocabulary its own ``optimizer`` key declares.
STOP_REASONS = ("cap", "rel_tol", "legacy_tol")
LBFGS_STOP_REASONS = ("converged", "maxfun", "maxiter", "linesearch", "abnormal")

#: The frozen stopping rule reads a step schedule, and a quasi-Newton run has
#: none, so the legacy readout is undefined for that arm. lbfgs_restart.py
#: writes these sentinels rather than None (an int64 column cannot hold None,
#: and the restart cache refuses object arrays under allow_pickle=False). The
#: gate pins them exactly, so no truncation-deficit statistic can ever read
#: them as a genuine step count and weight.
LBFGS_LEGACY_STOP = -1

#: The frozen protocol's cap, mirrored here rather than imported: a gate that
#: imports the code it validates cannot catch that code changing.
FROZEN_MAX_STEPS = 400

#: Slack on the order invariants. They hold exactly by construction (the same
#: float is copied, never recomputed), so this only absorbs a future rewrite
#: that recomputes a delta instead of carrying it.
SLACK = 1e-12

#: Wall-clock arrays: they differ on every run by construction, and they are
#: the only two arrays --compare-arrays is allowed to skip.
CLOCK_ARRAYS = ("restart_seconds", "elapsed_seconds")


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
#: shrinking a figure. The four converged blocks and the n = 16 catch-up (which
#: lands in ``pq``, taking it from 77 to 91) are the job list of
#: cloud/runner.py, itself fixed by REGISTRATION-CONVERGED.md.
#:
#: schema="converged" *requires* the ladder readout. optclass does not: two of
#: its three arms are L-BFGS, which never runs the instrumented Adam loop.
#: Those archives are still held to every converged invariant they do carry.
CORPUS = (
    Block("pq", "pq/pq_n*.npz", 91, restarts=200),
    Block("budget800", "budget800/pq_n*_sigma0.1.npz", 6, restarts=800),
    Block("ceiling_curve", "ceiling_curve/*/pq_n*_sigma0.1.npz", 17),
    Block("depth_ceiling", "depth_ceiling/*/pq_n*_sigma0.1.npz", 8, restarts=60),
    Block("shallow_peaking", "shallow_peaking/pq_n*_sigma0.1.npz", 6, restarts=60),
    Block("robustness", "robustness/*/pq_n*_sigma*.npz", 5, restarts=200),
    Block("step_budget", "step_budget/pq_n*_sigma0.1.npz", 13, restarts=16),
    Block("connectivity", "connectivity/conn_n*.npz", 23, schema="connectivity"),
    Block("converged", "converged/pq_n*_sigma0.1.npz", 90, restarts=32,
          schema="converged"),
    Block("converged_pilot", "converged_pilot/pq_n*_sigma0.1.npz", 5, restarts=16,
          schema="converged"),
    Block("lr_floor_sweep", "lr_floor_sweep/*/pq_n*_sigma0.1.npz", 3, restarts=16,
          schema="converged"),
    Block("optclass", "optclass/*/pq_n*_sigma*.npz", 45, restarts=16),
    # Exploratory, post-registration: SGD under the converged schedule at
    # n = 10, instance 0. Not part of any registered block or of the
    # runner's job list; gated here so the archive the manuscript quotes
    # is held to the same invariants as everything else.
    Block("sgd_converged", "sgd_converged/pq_n*_sigma0.1.npz", 1, restarts=16,
          schema="converged"),
)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_restart_archive(
    path: Path, restarts_expected: int | None, converged_required: bool = False
) -> list[str]:
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
        failures.extend(
            validate_converged_readout(data, restarts, deltas, converged_required)
        )
    return failures


def validate_converged_readout(
    data, restarts: int, peak_weights: np.ndarray, required: bool
) -> list[str]:
    """The ladder readout of a converged run, or nothing if it carries none.

    Every check here is on an invariant the writer guarantees by construction,
    so any failure means the archive and the protocol have drifted apart --
    which is exactly what a truncation correction computed from these arrays
    must not be allowed to do silently.
    """
    failures: list[str] = []
    for group, label in (
        (CONVERGED_SETTINGS_KEYS, "settings"),
        (CONVERGED_READOUT_KEYS, "readout"),
    ):
        missing = [key for key in group if key not in data.files]
        if missing and len(missing) != len(group):
            failures.append(
                f"converged {label} keys incomplete: missing {', '.join(missing)}"
            )
    if failures:
        return failures
    if not all(key in data.files for key in CONVERGED_READOUT_KEYS):
        if required:
            failures.append("no converged readout in a converged block")
        return failures
    if not all(key in data.files for key in CONVERGED_SETTINGS_KEYS):
        return ["converged readout without the settings that produced it"]

    max_steps = int(data["max_steps"])
    ladder = np.atleast_1d(data["ladder"])
    weights = data["ladder_weights"]
    reasons = data["stop_reasons"]
    running_max = data["running_max"]
    polish_gain = data["polish_gain"]
    legacy_weight = data["legacy_weight"]
    legacy_stop = data["legacy_stop_step"]
    num_steps = data["num_steps"]
    optimizer = str(data["optimizer"])
    lbfgs = optimizer == "lbfgs"

    # An L-BFGS arm runs without --converged, so it carries the readout with
    # no ladder at all; ladder_weights is then (restarts, 0) and the shape
    # check below is what holds it. A block that declares schema="converged"
    # is Adam and must have its rungs.
    check(
        not required or ladder.size > 0,
        "empty ladder in a converged archive",
        failures,
    )
    check(
        bool(np.all(np.diff(ladder) > 0)),
        f"ladder {ladder.tolist()} is not strictly increasing",
        failures,
    )
    check(
        ladder.size == 0 or int(ladder[-1]) <= max_steps,
        f"ladder ends at {int(ladder[-1]) if ladder.size else 0} > "
        f"max_steps {max_steps}",
        failures,
    )
    check(
        weights.shape == (restarts, ladder.size),
        f"ladder_weights shape {weights.shape}, expected "
        f"{(restarts, ladder.size)}",
        failures,
    )
    for key, array in (
        ("stop_reasons", reasons),
        ("running_max", running_max),
        ("polish_gain", polish_gain),
        ("legacy_stop_step", legacy_stop),
        ("legacy_weight", legacy_weight),
    ):
        check(len(array) == restarts, f"{key}: length mismatch", failures)
    vocabulary = LBFGS_STOP_REASONS if lbfgs else STOP_REASONS
    unknown = sorted(set(np.asarray(reasons).tolist()) - set(vocabulary))
    check(
        not unknown,
        f"unknown stop reason(s) {unknown} for optimizer {optimizer!r}",
        failures,
    )

    if weights.shape != (restarts, ladder.size) or len(running_max) != restarts:
        return failures  # shapes are wrong; the value checks would be noise

    for row in range(restarts):
        rung_weights = weights[row]
        recorded = ~np.isnan(rung_weights)
        # NaN means "the restart had already stopped", so it can only pad the
        # tail: a hole in the middle would be a lost rung, not an early stop.
        check(
            bool(np.all(np.diff(recorded.astype(np.int8)) <= 0)),
            f"restart {row}: ladder_weights has an interior NaN",
            failures,
        )
        filled = rung_weights[recorded]
        check(
            bool(np.all(np.diff(filled) >= -SLACK)),
            f"restart {row}: ladder_weights decreases along the ladder",
            failures,
        )
        if filled.size:
            check(
                filled[-1] <= running_max[row] + SLACK,
                f"restart {row}: last rung {filled[-1]:.6e} exceeds "
                f"running_max {running_max[row]:.6e}",
                failures,
            )
        if reasons[row] == "cap" and ladder.size and int(ladder[-1]) == max_steps:
            check(
                bool(recorded.all()),
                f"restart {row}: stopped at the cap with an unrecorded rung",
                failures,
            )

    check(
        bool(np.all(running_max <= peak_weights + SLACK)),
        f"running_max exceeds the reported peak weight on "
        f"{int(np.sum(running_max > peak_weights + SLACK))} restart(s)",
        failures,
    )
    check(
        bool(np.all(polish_gain >= -SLACK)),
        f"polish_gain negative on {int(np.sum(polish_gain < -SLACK))} restart(s)",
        failures,
    )
    check(bool(np.all(np.isfinite(running_max))), "non-finite running_max", failures)

    if lbfgs:
        # Undefined here, so the only thing to check is that it is the
        # sentinel and nothing else. A real step count in this arm would mean
        # the readout came from somewhere it could not have.
        check(
            bool(np.all(legacy_stop == LBFGS_LEGACY_STOP)),
            f"L-BFGS arm: legacy_stop_step is not {LBFGS_LEGACY_STOP} on "
            f"{int(np.sum(legacy_stop != LBFGS_LEGACY_STOP))} restart(s)",
            failures,
        )
        check(
            bool(np.all(np.isnan(legacy_weight))),
            f"L-BFGS arm: legacy_weight is not NaN on "
            f"{int(np.sum(~np.isnan(legacy_weight)))} restart(s)",
            failures,
        )
        return failures

    # Finiteness first: a NaN legacy_weight fails the ordering check below
    # while counting zero offending restarts, which reads as a phantom.
    check(
        bool(np.all(np.isfinite(legacy_weight))),
        f"non-finite legacy_weight on "
        f"{int(np.sum(~np.isfinite(legacy_weight)))} restart(s)",
        failures,
    )
    check(
        bool(np.all(legacy_weight <= running_max + SLACK)),
        f"legacy_weight exceeds running_max on "
        f"{int(np.sum(legacy_weight > running_max + SLACK))} restart(s)",
        failures,
    )
    # The frozen rule is replayed along the converged trajectory: it cannot
    # fire past the frozen cap, nor past the step the trajectory reached.
    ceiling = min(max_steps, FROZEN_MAX_STEPS)
    check(
        bool(np.all(legacy_stop >= 1) and np.all(legacy_stop <= ceiling)),
        f"legacy_stop_step outside [1, {ceiling}]",
        failures,
    )
    check(
        bool(np.all(legacy_stop <= num_steps)),
        "legacy_stop_step past the end of its own trajectory",
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


def array_difference(left: np.ndarray, right: np.ndarray) -> str | None:
    """None when the two arrays are bit-identical, else how far apart they are.

    The verdict is taken on the raw bytes, not on ``==``: NaN compares unequal
    to itself, and the ladder padding is NaN, so an equality test would report
    every converged archive as differing from its own copy.
    """
    if left.dtype != right.dtype:
        return f"dtype {left.dtype} vs {right.dtype}"
    if left.shape != right.shape:
        return f"shape {left.shape} vs {right.shape}"
    if left.tobytes() == right.tobytes():
        return None
    first, second = np.atleast_1d(left), np.atleast_1d(right)
    if not np.issubdtype(left.dtype, np.number):
        count = int(np.count_nonzero(first != second))
        return f"{count}/{first.size} entries differ"
    unequal = first != second
    if np.issubdtype(left.dtype, np.inexact):
        unequal &= ~(np.isnan(first) & np.isnan(second))
    count = int(np.count_nonzero(unequal))
    if not count:
        return "same values, different bytes (NaN payload or signed zero)"
    gap = np.abs(first[unequal] - second[unequal])
    return (f"{count}/{first.size} entries differ, max |delta| = "
            f"{float(gap.max()):.3e}")


def compare_archives(left: Path, right: Path) -> list[str]:
    """Every array but the two wall-clock ones must be bit-identical."""
    failures: list[str] = []
    with np.load(left) as first, np.load(right) as second:
        keys, other = set(first.files), set(second.files)
        for key in sorted(keys - other):
            failures.append(f"{key}: absent from the other archive")
        for key in sorted(other - keys):
            failures.append(f"{key}: present only in the other archive")
        for key in sorted(keys & other):
            if key in CLOCK_ARRAYS:
                continue
            difference = array_difference(first[key], second[key])
            if difference is not None:
                failures.append(f"{key}: {difference}")
    return failures


def compare_directories(results: Path, other: Path, quiet: bool) -> list[str]:
    """Binary-portability gate between two copies of the same corpus."""
    failures: list[str] = []
    names = sorted(
        {path.relative_to(root) for root in (results, other)
         for path in root.rglob("*.npz")}
    )
    compared = 0
    for name in names:
        left, right = results / name, other / name
        if not left.exists() or not right.exists():
            side = other if not right.exists() else results
            failures.append(f"{name}: missing under {side}")
            continue
        file_failures = compare_archives(left, right)
        compared += 1
        if not quiet:
            print(f"{str(name):<52} "
                  f"{'identical' if not file_failures else 'DIFFERS'}")
        failures.extend(f"{name}: {failure}" for failure in file_failures)
    print(f"\n{compared} archive(s) compared array by array, "
          f"{', '.join(CLOCK_ARRAYS)} excluded as wall-clock")
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
    parser.add_argument("--compare-arrays", type=str, default=None,
                        metavar="OTHER_DIR",
                        help="compare every array but restart_seconds and "
                             "elapsed_seconds against a second copy of the "
                             "corpus, and report what moved")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def report(failures: list[str]) -> None:
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("all checks passed")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = Path(args.results)
    if not results.is_absolute():
        results = ROOT / results

    if args.compare_arrays is not None:
        other = Path(args.compare_arrays)
        if not other.is_absolute():
            other = ROOT / other
        report(compare_directories(results, other, args.quiet))
        return

    failures: list[str] = []
    found = collect(results)

    counts: dict[str, int] = {}
    for block, path in found:
        counts[block.name] = counts.get(block.name, 0) + 1
        if block.schema == "connectivity":
            file_failures = validate_connectivity_archive(path)
        else:
            file_failures = validate_restart_archive(
                path, block.restarts, converged_required=block.schema == "converged"
            )
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
        print(f"\nmanifest: {checked}/{len(recorded)} archives pinned byte for "
              "byte to their recorded digest")
    else:
        print(f"\nno {MANIFEST} present; run --write-manifest to create one")

    total = sum(block.files for block in CORPUS)
    print(f"{len(found)}/{total} archive(s) validated")
    report(failures)


if __name__ == "__main__":
    main()
