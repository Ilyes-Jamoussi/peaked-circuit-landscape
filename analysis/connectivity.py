"""Connectivity test between saved solutions.

For pairs of near-optimal solutions of the same instance, measures delta
along (i) the raw parameter segment and (ii) a string-method re-optimized
path (endpoints fixed; alternate independent Adam ascent on interior
waypoints and equal-arc-length reparameterization). Reports the minimum
of delta over a dense grid along the final polyline (interior only), the
gauge-invariant overlap curves q(psi_t, psi_A/B), and path validity.

Scrambled controls pair a solution of instance 0 with a solution of
instance 1 evaluated on instance 0's field (typical delta endpoint): the
string must not fake a near-optimal-level path to a typical point.

All verdict criteria are registered BEFORE any run.

Usage:
    python analysis/connectivity.py --num-qubits 8
    python analysis/connectivity.py --num-qubits 8 --budget-factor 4 --pairs 0,3,6,9
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
     segments) = job
    config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
    layers = sample_haar_random_layers(
        config,
        np.random.default_rng(np.random.SeedSequence(42, spawn_key=(num_qubits, 0))),
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


def select_jobs(num_qubits: int, budget_factor: int, pair_filter) -> list:
    data0 = np.load(RESULTS / "pq" / f"pq_n{num_qubits}_i0_sigma0.1.npz")
    data1 = np.load(RESULTS / "pq" / f"pq_n{num_qubits}_i1_sigma0.1.npz")
    thetas0, deltas0 = data0["thetas_final"], data0["peak_weights"]
    thetas1, deltas1 = data1["thetas_final"], data1["peak_weights"]

    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED, spawn_key=(num_qubits,)))
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
             BASE_SEED, NUM_SEGMENTS)
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
            )
        )
    return jobs


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
    print("self-tests passed (reparameterization, dense grid)")


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
    args = parser.parse_args()

    self_tests()
    pair_filter = (
        {int(x) for x in args.pairs.split(",")} if args.pairs else None
    )
    jobs = select_jobs(args.num_qubits, args.budget_factor, pair_filter)
    if args.outer is not None or args.steps is not None or args.segments is not None:
        jobs = [
            (kind, index, nq, ta, tb,
             args.outer if args.outer is not None else outer,
             args.steps if args.steps is not None else steps, seed,
             args.segments if args.segments is not None else segments)
            for (kind, index, nq, ta, tb, outer, steps, seed, segments) in jobs
        ]
    dim = 2.0 ** args.num_qubits
    print(
        f"n = {args.num_qubits}: {len(jobs)} jobs "
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
                f" [{r['seconds']:.0f} s]"
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
    suffix = f"_x{args.budget_factor}" if args.budget_factor > 1 else ""
    if args.outer is not None or args.steps is not None:
        suffix += f"_o{jobs[0][5]}s{jobs[0][6]}"
    if args.segments is not None:
        suffix += f"_m{jobs[0][8]}"
    path = out_dir / f"conn_n{args.num_qubits}{suffix}.npz"
    np.savez_compressed(path, **stored)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
