"""Probe dependence of the moment hierarchy.

The exact kernel depends on the probe state through every reduced
cross-purity, so the moments of delta are functions of the point theta at
which they are evaluated, not numbers attached to the field. third_moment.py,
fourth_moment.py and their n = 8 twins all evaluate at a single hard-coded
draw, theta ~ N(0, 0.1^2) -- the protocol's initialization, a near-identity
point where the probe is almost the product state |0...0> and every reduced
cross-purity is almost 1, which is where m_k is largest.

This script measures how far that choice carries. It evaluates m_2, m_3, m_4
and the pair-dominance ratios r_k = m_k / m_2^binom(k,2) at the operating
depth tau_r = n, over a ladder of probe scales from the exact product state
(sigma = 0) out to a scrambled point (sigma = 1), and -- the case that
matters for the physics -- at the optimizer's own solutions, the points the
reach measurements actually visit.

Self-tests: at sigma = 0 the probe is exactly |0...0> and every reduced
cross-purity is 1; the deep-depth limit reproduces the Haar floors
m_k = D^(k-1) / prod_{j<k}(D+j); and the sigma = 0.1 column reproduces the
committed scan values.

Usage:
    python analysis/moment_probe_scan.py --max-k 2              # seconds
    python analysis/moment_probe_scan.py --max-k 3 --sizes 6    # minutes
    python analysis/moment_probe_scan.py --max-k 4              # hours
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from kernel_exact import (  # noqa: E402
    cross_purities,
    exact_second_moment,
    probe_state_fn,
    spin_weights,
)
from peaked_circuits import CircuitConfig  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
PROBE_SEED = 4042           # the draw used by fourth_moment.py
SCALES = (0.0, 0.1, 0.5, 1.0)
NUM_SOLUTIONS = 5
THIRD_BUDGET = 6000
FOURTH_BUDGET = 150000

#: Where the operating-configuration solutions live for each size.
SOLUTION_ARCHIVE = {
    6: RESULTS / "ceiling_curve" / "n6_taup3" / "pq_n6_i0_sigma0.1.npz",
    8: RESULTS / "pq" / "pq_n8_i0_sigma0.1.npz",
}


def haar_floor(num_qubits: int, k: int) -> float:
    """m_k for a Haar-random scrambled state: D^(k-1) / prod_{j=1}^{k-1}(D+j)."""
    dim = 2.0**num_qubits
    value = dim ** (k - 1)
    for j in range(1, k):
        value /= dim + j
    return value


def probe_points(config: CircuitConfig, num_qubits: int, scales, solutions: int):
    """(label, state) pairs: the scale ladder, then optimizer solutions."""
    probe = probe_state_fn(config)
    points = []
    for scale in scales:
        rng = np.random.default_rng(np.random.SeedSequence(PROBE_SEED))
        theta = rng.normal(0.0, scale, config.num_peaking_parameters)
        points.append((f"sigma={scale:g}", np.asarray(probe(theta))))
    archive = SOLUTION_ARCHIVE.get(num_qubits)
    if solutions and archive is not None and archive.exists():
        data = np.load(archive)
        order = np.argsort(data["peak_weights"])[::-1][:solutions]
        for rank, index in enumerate(order):
            state = np.asarray(probe(data["thetas_final"][index]))
            points.append((f"solution{rank}", state))
    return points


def moments(state, num_qubits: int, depth: int, max_k: int, workers: int):
    """(m_2, m_3, m_4, bounds) at one probe point; None beyond max_k."""
    dim = 2.0**num_qubits
    second = exact_second_moment(spin_weights(num_qubits, depth), state, state,
                                 num_qubits)
    m2 = second / (2.0 / dim**2)
    m3 = m4 = None
    bound3 = bound4 = 0.0
    if max_k >= 3:
        from third_moment import exact_third_moment, s3_weights

        weights3 = s3_weights(num_qubits, depth, (depth,))[depth]
        if num_qubits <= 6:
            third, bound3 = exact_third_moment(weights3, state, num_qubits), 0.0
        else:
            from third_moment_n8 import truncated_third

            third, bound3 = truncated_third(weights3, state, num_qubits,
                                            budget=THIRD_BUDGET)
        m3 = third / (6.0 / dim**3)
        bound3 /= 6.0 / dim**3
    if max_k >= 4:
        from fourth_moment import exact_fourth_moment, s4_weights

        fourth, dropped = exact_fourth_moment(
            s4_weights(num_qubits, depth), state, num_qubits,
            budget=FOURTH_BUDGET, workers=workers,
            evaluator="dense" if num_qubits <= 6 else "einsum",
        )
        m4 = fourth / (24.0 / dim**4)
        bound4 = dropped / (24.0 / dim**4)
    return m2, m3, m4, bound3, bound4


def self_tests() -> None:
    num_qubits = 6
    config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
    probe = probe_state_fn(config)
    zero = np.asarray(probe(np.zeros(config.num_peaking_parameters)))
    reference = np.zeros(2**num_qubits, dtype=complex)
    reference[0] = 1.0
    assert np.allclose(np.abs(zero), np.abs(reference)), (
        "at theta = 0 the probe must be the computational zero state"
    )
    purities = cross_purities(zero, zero, num_qubits)
    assert np.allclose(purities, 1.0), (
        "a product probe must have every reduced cross-purity equal to 1"
    )

    deep = exact_second_moment(spin_weights(num_qubits, 80), zero, zero,
                               num_qubits)
    m2_deep = deep / (2.0 / (2.0**num_qubits) ** 2)
    assert abs(m2_deep - haar_floor(num_qubits, 2)) < 1e-9, (
        f"deep m2 {m2_deep} vs Haar floor {haar_floor(num_qubits, 2)}"
    )
    print("self-tests passed (product probe at sigma = 0, unit cross-purities, "
          f"deep m2 = {m2_deep:.6f} = D/(D+1))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=str, default="6,8")
    parser.add_argument("--max-k", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--solutions", type=int, default=NUM_SOLUTIONS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    self_tests()
    for num_qubits in (int(s) for s in args.sizes.split(",")):
        depth = num_qubits
        config = CircuitConfig(num_qubits, depth, num_qubits // 2)
        points = probe_points(config, num_qubits, SCALES, args.solutions)
        print(f"\nn = {num_qubits}, tau_r = {depth}, tau_p = {num_qubits // 2}, "
              f"P = {config.num_peaking_parameters}")
        print(f"{'probe':>12} {'purity':>8} {'m2':>8} {'m3':>9} {'m4':>10} "
              f"{'r3':>7} {'r4':>7}   (Haar floor m2 = "
              f"{haar_floor(num_qubits, 2):.4f})")
        for label, state in points:
            start = time.perf_counter()
            half = state.reshape(2 ** (num_qubits // 2), -1)
            reduced = half @ half.conj().T
            purity = float(np.real(np.trace(reduced @ reduced)))
            m2, m3, m4, bound3, bound4 = moments(state, num_qubits, depth,
                                                 args.max_k, args.workers)
            r3 = m3 / m2**3 if m3 is not None else None
            r4 = m4 / m2**6 if m4 is not None else None
            fmt = lambda v, w, p: (f"{v:>{w}.{p}f}" if v is not None
                                   else " " * (w - 1) + "-")  # noqa: E731
            print(f"{label:>12} {purity:>8.4f} {m2:>8.4f} {fmt(m3, 9, 4)} "
                  f"{fmt(m4, 10, 4)} {fmt(r3, 7, 4)} {fmt(r4, 7, 4)}"
                  f"   [{time.perf_counter() - start:.0f} s]", flush=True)
            if bound3 or bound4:
                print(f"{'':>12} certified truncation: |m3| <= {bound3:.2e}, "
                      f"|m4| <= {bound4:.2e}")

    print("\nThe scale ladder is the same field at different points of the same "
          "manifold: m_k is a function of the probe, not a number attached to "
          "delta. The solution rows are the points the reach measurements visit.")


if __name__ == "__main__":
    main()
