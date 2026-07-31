"""Connectivity test between saved solutions.

For pairs of near-optimal solutions of the same instance, measures delta
along (i) the raw parameter segment and (ii) a string-method re-optimized
path (endpoints fixed; alternate independent Adam ascent on interior
waypoints and equal-arc-length reparameterization). Reports the minimum
of delta over a dense grid along the final polyline (interior only), the
gauge-invariant overlap curves q(psi_t, psi_A/B), and path validity.

Scrambled controls pair a solution of the measured instance with a
solution of another instance evaluated on the first one's field (typical
delta endpoint): the string must not fake a near-optimal-level path to a
typical point.

All verdict criteria are registered BEFORE any run.

Usage:
    python analysis/connectivity.py --num-qubits 8
    python analysis/connectivity.py --num-qubits 8 --budget-factor 4 --pairs 0,3,6,9
    python analysis/connectivity.py --num-qubits 12 --instance 1 \
        --outer 150 --steps 1 --segments 64      # instance sweep, matched resolution
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

NUM_PAIRS = 12
NUM_CONTROLS = 4
NUM_SEGMENTS = 16
OUTER_ITERATIONS = 30
STEPS_PER_WAYPOINT = 5
LEARNING_RATE = 0.02
DENSE_PER_SEGMENT = 8
EPSILON = 0.2
BASE_SEED = 9099
RESULTS = Path(__file__).resolve().parent.parent / "results"


def polyline_length(waypoints: np.ndarray) -> np.ndarray:
    """Cumulative arc length of a (M+1, P) polyline, starting at 0."""
    steps = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def reparameterize(waypoints: np.ndarray) -> np.ndarray:
    """Resample the polyline at equal arc-length spacing; endpoints fixed."""
    cumulative = polyline_length(waypoints)
    total = cumulative[-1]
    if total == 0.0:
        return waypoints.copy()
    targets = np.linspace(0.0, total, len(waypoints))
    resampled = np.empty_like(waypoints)
    for coord in range(waypoints.shape[1]):
        resampled[:, coord] = np.interp(targets, cumulative, waypoints[:, coord])
    resampled[0], resampled[-1] = waypoints[0], waypoints[-1]
    return resampled


def dense_grid(waypoints: np.ndarray, per_segment: int) -> np.ndarray:
    """Dense sample points along the polyline (includes both endpoints)."""
    points = []
    for k in range(len(waypoints) - 1):
        for j in range(per_segment):
            t = j / per_segment
            points.append((1 - t) * waypoints[k] + t * waypoints[k + 1])
    points.append(waypoints[-1])
    return np.stack(points)


def run_job(job) -> dict:
    """One (pair or control) string optimization; executed in a worker."""
    import pennylane as qml  # noqa: F401  (worker import)
    from pennylane import numpy as pnp

    import pq_experiment  # noqa: F401
    from peaked_circuits import (
        CircuitConfig,
        build_peak_weight_fn,
        sample_haar_random_layers,
    )
    from pq_experiment import build_state_fn

    (kind, index, num_qubits, theta_a, theta_b, outer, steps, seed,
     segments, instance) = job
    config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
    # The field is always the measured instance's, for controls too: a
    # control walks toward a foreign solution *on this instance's field*.
    layers = sample_haar_random_layers(
        config,
        np.random.default_rng(
            np.random.SeedSequence(42, spawn_key=(num_qubits, instance))
        ),
    )
    field = build_peak_weight_fn(config, layers)
    state_fn = build_state_fn(config, layers)

    def cost_factory():
        def cost(parameters):
            return -field(parameters)

        return cost

    start = time.perf_counter()
    waypoints = np.linspace(0.0, 1.0, segments + 1)[:, None] * (
        theta_b - theta_a
    ) + theta_a
    initial_length = float(polyline_length(waypoints)[-1])

    raw_values = np.array([float(field(p)) for p in dense_grid(waypoints, DENSE_PER_SEGMENT)])

    for _ in range(outer):
        for k in range(1, segments):
            theta = pnp.array(waypoints[k], requires_grad=True)
            optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)
            cost = cost_factory()
            for _ in range(steps):
                theta, _ = optimizer.step_and_cost(cost, theta)
            waypoints[k] = np.asarray(theta)
        waypoints = reparameterize(waypoints)

    dense = dense_grid(waypoints, DENSE_PER_SEGMENT)
    values = np.array([float(field(p)) for p in dense])
    final_length = float(polyline_length(waypoints)[-1])
    spacing = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

    states = np.stack([np.asarray(state_fn(p)) for p in waypoints])
    overlap_a = np.abs(states @ states[0].conj()) ** 2
    overlap_b = np.abs(states @ states[-1].conj()) ** 2
    adjacent = np.abs(np.sum(states[1:] * states[:-1].conj(), axis=1)) ** 2

    return dict(
        kind=kind,
        index=index,
        delta_a=float(field(theta_a)),
        delta_b=float(field(theta_b)),
        raw_values=raw_values,
        values=values,
        min_interior=float(values[1:-1].min()),
        raw_min_interior=float(raw_values[1:-1].min()),
        initial_length=initial_length,
        final_length=final_length,
        max_spacing=float(spacing.max()),
        mean_spacing=float(spacing.mean()),
        overlap_a=overlap_a,
        overlap_b=overlap_b,
        min_adjacent_overlap=float(adjacent.min()),
        waypoints=waypoints,
        seconds=time.perf_counter() - start,
    )


def select_jobs(num_qubits: int, budget_factor: int, pair_filter,
                instance: int = 0, control_instance: int | None = None) -> list:
    """Pair and control jobs for one (size, instance).

    `instance` supplies the near-optimal solutions that are paired; the
    scrambled controls pair one of them with a solution of
    `control_instance` (default: the next instance) evaluated on
    `instance`'s field.
    """
    if control_instance is None:
        control_instance = instance + 1
    if control_instance == instance:
        raise ValueError("control partner must come from a different instance")
    data0 = np.load(RESULTS / "pq" / f"pq_n{num_qubits}_i{instance}_sigma0.1.npz")
    data1 = np.load(
        RESULTS / "pq" / f"pq_n{num_qubits}_i{control_instance}_sigma0.1.npz"
    )
    thetas0, deltas0 = data0["thetas_final"], data0["peak_weights"]
    thetas1, deltas1 = data1["thetas_final"], data1["peak_weights"]

    # Instance 0 keeps the original spawn key so that every archive
    # committed before the instance sweep regenerates unchanged; the added
    # instances extend the key rather than shifting it.
    spawn_key = (num_qubits,) if instance == 0 else (num_qubits, instance)
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED, spawn_key=spawn_key))
    near0 = np.flatnonzero(deltas0 >= (1 - EPSILON) * deltas0.max())
    near1 = np.flatnonzero(deltas1 >= (1 - EPSILON) * deltas1.max())

    chosen = rng.choice(near0, size=2 * NUM_PAIRS + NUM_CONTROLS, replace=False)
    partners = rng.choice(near1, size=NUM_CONTROLS, replace=False)

    if budget_factor == 1:
        outer, steps = OUTER_ITERATIONS, STEPS_PER_WAYPOINT
    elif budget_factor == 4:
        outer, steps = OUTER_ITERATIONS * 2, STEPS_PER_WAYPOINT * 2
    else:
        raise ValueError("budget_factor must be 1 or 4 (registered protocol)")

    jobs = []
    for pair in range(NUM_PAIRS):
        if pair_filter is not None and pair not in pair_filter:
            continue
        a, b = chosen[2 * pair], chosen[2 * pair + 1]
        jobs.append(
            ("pair", pair, num_qubits, thetas0[a], thetas0[b], outer, steps,
             BASE_SEED, NUM_SEGMENTS, instance)
        )
    for control in range(NUM_CONTROLS):
        if pair_filter is not None:
            continue
        a = chosen[2 * NUM_PAIRS + control]
        jobs.append(
            (
                "control",
                control,
                num_qubits,
                thetas0[a],
                thetas1[partners[control]],
                outer,
                steps,
                BASE_SEED,
                NUM_SEGMENTS,
                instance,
            )
        )
    return jobs


def resolved_schedule(args) -> tuple[int, int, int]:
    """(outer, steps, segments) after the budget factor and CLI overrides."""
    if args.budget_factor == 1:
        outer, steps = OUTER_ITERATIONS, STEPS_PER_WAYPOINT
    elif args.budget_factor == 4:
        outer, steps = OUTER_ITERATIONS * 2, STEPS_PER_WAYPOINT * 2
    else:
        raise ValueError("budget_factor must be 1 or 4 (registered protocol)")
    if args.outer is not None:
        outer = args.outer
    if args.steps is not None:
        steps = args.steps
    segments = args.segments if args.segments is not None else NUM_SEGMENTS
    return outer, steps, segments


def archive_path(args) -> Path:
    """Output archive for this invocation (single source of truth)."""
    outer, steps, segments = resolved_schedule(args)
    # Instance 0 keeps its historical file names (no _i tag), so the
    # archives committed before the instance sweep keep resolving.
    instance_tag = f"_i{args.instance}" if args.instance else ""
    suffix = f"_x{args.budget_factor}" if args.budget_factor > 1 else ""
    if args.outer is not None or args.steps is not None:
        suffix += f"_o{outer}s{steps}"
    if args.segments is not None:
        suffix += f"_m{segments}"
    return (RESULTS / "connectivity"
            / f"conn_n{args.num_qubits}{instance_tag}{suffix}.npz")


def guard_existing_archive(args) -> None:
    """Refuse to replace a stored archive unless --force is given.

    A --pairs subset writes only the pairs it ran, so an unguarded rerun
    silently replaces a full archive with a partial one.
    """
    path = archive_path(args)
    if path.exists() and not args.force:
        raise SystemExit(
            f"{path.name} already exists in results/connectivity/.\n"
            "Rerunning would replace it, and a --pairs subset would replace a "
            "complete archive with a partial one. Pass --force to overwrite."
        )


def self_tests() -> None:
    # Reparameterization is equal-spacing in arc length along the OLD
    # polyline; chord distances of the new waypoints can differ at
    # corners, so the check is tolerant (and exact on a straight line).
    line = np.linspace(0.0, 1.0, 5)[:, None] * np.ones((1, 3))
    straight = reparameterize(line * 2.0)
    spacing = np.linalg.norm(np.diff(straight, axis=0), axis=1)
    assert spacing.std() / spacing.mean() < 1e-12, "straight-line spacing"
    warped = line.copy()
    warped[2] = [0.9, 0.9, 0.9]
    fixed = reparameterize(warped)
    assert np.allclose(fixed[0], line[0]) and np.allclose(fixed[-1], line[-1])
    spacing = np.linalg.norm(np.diff(fixed, axis=0), axis=1)
    assert spacing.max() <= 2.0 * spacing.mean(), "waypoint continuity"
    grid = dense_grid(line, 4)
    assert np.allclose(grid[0], line[0]) and np.allclose(grid[-1], line[-1])
    assert len(grid) == 4 * 4 + 1

    # Instance wiring: a job's endpoints must reproduce the peakedness
    # stored for THEIR instance. This catches a field built for the wrong
    # instance, which is silent otherwise (the endpoints simply look
    # typical, delta ~ 2^-n, instead of near-optimal).
    import pq_experiment  # noqa: F401  (wires up the reproduction repo path)
    from peaked_circuits import (
        CircuitConfig,
        build_peak_weight_fn,
        sample_haar_random_layers,
    )

    for instance in (0, 1):
        job = select_jobs(8, 1, {0}, instance=instance)[0]
        config = CircuitConfig(8, 8, 4)
        layers = sample_haar_random_layers(
            config,
            np.random.default_rng(
                np.random.SeedSequence(42, spawn_key=(8, job[9]))
            ),
        )
        field = build_peak_weight_fn(config, layers)
        stored = np.load(RESULTS / "pq" / f"pq_n8_i{instance}_sigma0.1.npz")
        best = float(stored["peak_weights"].max())
        for endpoint in (job[3], job[4]):
            value = float(field(endpoint))
            assert value >= (1 - EPSILON) * best, (
                f"instance {instance}: endpoint evaluates to {value:.4g}, "
                f"below the near-optimal filter {(1 - EPSILON) * best:.4g} "
                "-- the field is probably built for the wrong instance"
            )
    print("self-tests passed (reparameterization, dense grid, instance wiring)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--budget-factor", type=int, default=1)
    parser.add_argument("--pairs", type=str, default=None,
                        help="comma-separated pair indices (budget rerun)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--outer", type=int, default=None,
                        help="override outer iterations (post-hoc schedules)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override ascent steps per waypoint")
    parser.add_argument("--segments", type=int, default=None,
                        help="override waypoint segments (post-hoc resolution)")
    parser.add_argument("--instance", type=int, default=0,
                        help="instance whose near-optimal solutions are paired")
    parser.add_argument("--control-instance", type=int, default=None,
                        help="instance supplying scrambled-control partners "
                             "(default: --instance + 1)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing archive (refused by "
                             "default: a --pairs subset would replace a full "
                             "archive with a partial one)")
    args = parser.parse_args()

    self_tests()
    guard_existing_archive(args)
    pair_filter = (
        {int(x) for x in args.pairs.split(",")} if args.pairs else None
    )
    jobs = select_jobs(args.num_qubits, args.budget_factor, pair_filter,
                       instance=args.instance,
                       control_instance=args.control_instance)
    if args.outer is not None or args.steps is not None or args.segments is not None:
        jobs = [
            (kind, index, nq, ta, tb,
             args.outer if args.outer is not None else outer,
             args.steps if args.steps is not None else steps, seed,
             args.segments if args.segments is not None else segments,
             instance)
            for (kind, index, nq, ta, tb, outer, steps, seed, segments,
                 instance) in jobs
        ]
    dim = 2.0 ** args.num_qubits
    print(
        f"n = {args.num_qubits}, instance {args.instance}: {len(jobs)} jobs "
        f"(budget x{args.budget_factor}, {jobs[0][8]} segments)"
    )

    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=__import__("multiprocessing").get_context("spawn"),
    ) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            threshold = 0.8 * (
                min(r["delta_a"], r["delta_b"])
                if r["kind"] == "pair"
                else r["delta_a"]
            )
            passes = r["min_interior"] >= threshold
            valid = r["max_spacing"] <= 2.0 * r["mean_spacing"] + 1e-12
            print(
                f"{r['kind']:>8} {r['index']:>2}: dA={r['delta_a']:.3f} "
                f"dB={r['delta_b']:.3f}  raw_min={r['raw_min_interior']:.2e} "
                f" string_min={r['min_interior']:.4f}  thr={threshold:.3f} "
                f" {'PASS' if passes else 'fail'}{'' if valid else ' INVALID'} "
                f" len x{r['final_length'] / r['initial_length']:.2f} "
                f" min_adj_q={r['min_adjacent_overlap']:.3f} "
                f" [{r['seconds']:.0f} s]",
                flush=True,   # progress must be visible in a redirected log
            )

    pair_results = [r for r in results if r["kind"] == "pair"]
    control_results = [r for r in results if r["kind"] == "control"]
    if pair_results:
        passing = [
            r
            for r in pair_results
            if r["min_interior"] >= 0.8 * min(r["delta_a"], r["delta_b"])
            and r["max_spacing"] <= 2.0 * r["mean_spacing"] + 1e-12
        ]
        print(
            f"\npairs passing C1: {len(passing)}/{len(pair_results)}"
            f"  (raw-segment medians: min = "
            f"{np.median([r['raw_min_interior'] for r in pair_results]):.2e},"
            f" typical scale 2^-n = {1 / dim:.2e})"
        )
    if control_results:
        broken = [
            r for r in control_results if r["min_interior"] >= 0.8 * r["delta_a"]
        ]
        minima = ", ".join(f"{r['min_interior']:.2e}" for r in control_results)
        verdict = "BROKEN" if broken else "sane"
        print(f"controls: interior minima = [{minima}] -> instrument {verdict}")

    out_dir = RESULTS / "connectivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    stored = {}
    for r in results:
        tag = f"{r['kind']}{r['index']}"
        for key in ("values", "raw_values", "overlap_a", "overlap_b", "waypoints"):
            stored[f"{tag}_{key}"] = r[key]
        stored[f"{tag}_scalars"] = np.array(
            [r["delta_a"], r["delta_b"], r["min_interior"], r["raw_min_interior"],
             r["initial_length"], r["final_length"], r["min_adjacent_overlap"]]
        )
    path = archive_path(args)
    np.savez_compressed(path, **stored)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
