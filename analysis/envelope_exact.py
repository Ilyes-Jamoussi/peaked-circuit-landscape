"""Exact anatomy of the multiplicative envelope.

Under the model delta = s * e (envelope times Porter-Thomas), the envelope
two-point function is the exact ratio
E[ss'] = E[dd']_exact / E[dd']_PT(F), computable at any pair by the kernel
machinery. This script scans the envelope covariance C(t) = ratio - 1
along parameter-space separations, decomposes it into an instance
component (the plateau read off the far tail of the ray, F < 0.01 --
a second near-identity draw is NOT decorrelated and cannot serve as the
plateau; corrected 2026-07-16) and a landscape component (short-range
excess, correlation scale xi), for n = 6..12 at tau_r = n.

Mandatory consistencies: C(0) = m2 (D+1)/D - 1, and C -> 0 at deep tau_r.

Usage:
    python analysis/envelope_exact.py       # ~10 min, writes envelope_exact.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from kernel_exact import (  # noqa: E402
    exact_second_moment,
    probe_state_fn,
    spin_weights,
)
from peaked_circuits import CircuitConfig  # noqa: E402

QUBIT_COUNTS = (6, 8, 10, 12)
SHIFTS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.5)
BASE_SEED = 5055
PROBE_SCALE = 0.1


def pt_second_moment(overlap: float, num_qubits: int) -> float:
    dim = 2.0**num_qubits
    return (1.0 + overlap) / (dim * (dim + 1.0))


def envelope_covariance(
    weights, state_a, state_b, num_qubits: int
) -> tuple[float, float]:
    """Returns (F, C) for one pair."""
    overlap = float(np.abs(np.vdot(state_a, state_b)) ** 2)
    exact = exact_second_moment(weights, state_a, state_b, num_qubits)
    return overlap, exact / pt_second_moment(overlap, num_qubits) - 1.0


def main() -> None:
    fig, axes = plt.subplots(
        1, len(QUBIT_COUNTS), figsize=(3.1 * len(QUBIT_COUNTS), 3.2), sharey=True
    )
    summary = {}
    for axis, num_qubits in zip(axes, QUBIT_COUNTS, strict=True):
        start = time.perf_counter()
        dim = 2.0**num_qubits
        config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
        num_parameters = config.num_peaking_parameters
        probe = probe_state_fn(config)
        rng = np.random.default_rng(
            np.random.SeedSequence(BASE_SEED, spawn_key=(num_qubits,))
        )
        theta = rng.normal(0.0, PROBE_SCALE, num_parameters)
        base_state = np.asarray(probe(theta))
        direction = rng.normal(0.0, 1.0, num_parameters)
        direction /= np.linalg.norm(direction)

        weights = spin_weights(num_qubits, num_qubits)

        variance_row = envelope_covariance(
            weights, base_state, base_state, num_qubits
        )
        c_zero = variance_row[1]

        rows = []
        for t in SHIFTS:
            shifted = np.asarray(
                probe(theta + t * np.sqrt(num_parameters) * direction)
            )
            overlap, cov = envelope_covariance(
                weights, base_state, shifted, num_qubits
            )
            rows.append((t, overlap, cov))

        # Instance component = C at probe-decorrelated separations. A second
        # draw at the near-identity scale is NOT decorrelated (F ~ 0.2-0.5),
        # so the plateau is read off the far tail of the ray (F < 0.01);
        # the independent draw is kept as a diagnostic only. (Corrected
        # 2026-07-16: the first version used the independent draw as the
        # plateau, overestimating it and shrinking xi by ~2x.)
        tail = [cov for _, overlap, cov in rows if overlap < 0.01]
        plateau = float(np.mean(tail)) if tail else rows[-1][2]
        independent = np.asarray(
            probe(rng.normal(0.0, PROBE_SCALE, num_parameters))
        )
        overlap_ind, c_independent = envelope_covariance(
            weights, base_state, independent, num_qubits
        )

        deep_weights = spin_weights(num_qubits, 4 * num_qubits)
        _, deep_c = envelope_covariance(
            weights=deep_weights,
            state_a=base_state,
            state_b=np.asarray(
                probe(theta + 0.3 * np.sqrt(num_parameters) * direction)
            ),
            num_qubits=num_qubits,
        )

        sigma2_land = c_zero - plateau
        half = plateau + sigma2_land / 2.0
        xi = next(
            (
                t * np.sqrt(num_parameters)
                for t, _, cov in rows
                if cov <= half
            ),
            float("nan"),
        )
        summary[num_qubits] = dict(
            c_zero=c_zero, plateau=plateau, sigma2=sigma2_land, xi=xi,
            num_parameters=num_parameters,
        )

        print(
            f"n = {num_qubits}: C(0) = {c_zero:.4f} "
            f"(m2 check: {(c_zero + 1) * dim / (dim + 1):.4f}), "
            f"plateau (instance, ray tail F<0.01) = {plateau:+.5f}, "
            f"landscape sigma^2 = {sigma2_land:.4f}, "
            f"xi = {xi:.2f} (param distance), deep check C = {deep_c:+.1e}, "
            f"second-draw diagnostic C = {c_independent:+.4f} at F = "
            f"{overlap_ind:.1e}  [{time.perf_counter() - start:.0f} s]"
        )
        for t, overlap, cov in rows:
            print(f"    t = {t:4.2f}  F = {overlap:7.4f}  C = {cov:+.5f}")

        ts = [row[0] for row in rows]
        cs = [row[2] for row in rows]
        axis.plot(ts, cs, "o-", markersize=4)
        axis.axhline(plateau, color="#d62728", linestyle="--", linewidth=1)
        axis.axhline(0.0, color="0.6", linewidth=0.7)
        axis.set_title(f"n = {num_qubits}", fontsize=10)
        axis.set_xlabel("t")
        axis.set_xscale("log")
    axes[0].set_ylabel("envelope covariance $C(t)$")
    fig.suptitle(
        "Envelope covariance vs separation (dashed: instance plateau)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).resolve().parent / "envelope_exact.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    print("\nsummary:")
    for num_qubits, values in summary.items():
        print(
            f"  n = {num_qubits}: sigma2_landscape = {values['sigma2']:.4f}, "
            f"instance = {values['plateau']:+.5f}, xi = {values['xi']:.2f}, "
            f"P = {values['num_parameters']}"
        )


if __name__ == "__main__":
    main()
