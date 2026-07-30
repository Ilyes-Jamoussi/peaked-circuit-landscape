"""MPS ceiling B_MPS on delta* by alternating least squares (09, Section 4).

The probe manifold is contained in MPS(r) with bond profile
r_b = min(2^g(b), 2^b, 2^(n-b)), so delta* <= B_MPS <= B_min-cut.
B_MPS = max over MPS(r) of |<m|phi>|^2 is computed by ALS: sweeps of
exact single-site updates (the optimal unit-norm tensor is the
normalized environment contraction, in mixed-canonical form),
initialized from the SVD truncation of phi plus random restarts. Any
ALS value below a measured delta certifies ALS non-convergence, never
a bound violation (delta* <= B_MPS is a theorem).

Self-tests: exact recovery of a random MPS target at its own bond
profile, overlap 1 at unrestricted bond, monotonicity in bond
dimension, and ALS <= B_min-cut.

Usage:
    python analysis/mps_ceiling.py     # ~2 min
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from ceiling_bound import bound_for_state, crossing_counts, scrambled_state  # noqa: E402
from peaked_circuits import CircuitConfig  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
SWEEPS = 200
RESTARTS = 6
TOL = 1e-13


def bond_profile(config: CircuitConfig) -> list[int]:
    # CORRECTION 2026-07-16: operator Schmidt rank of a generic two-qubit
    # gate is 4, so bonds grow as 4^g, not 2^g (see ceiling_bound.py).
    # At tau_p = n/2 the profile saturates every cut: the MPS class is
    # the full Hilbert space and B_MPS = 1 (vacuous).
    counts = crossing_counts(config)
    n = config.num_qubits
    return [int(min(4 ** counts[b], 2**b, 2 ** (n - b))) for b in range(1, n)]


def truncation_init(state: np.ndarray, bonds: list[int]) -> list[np.ndarray]:
    """Left-to-right SVD compression of the state to the bond profile."""
    n = len(bonds) + 1
    tensors = []
    remainder = state.reshape(1, -1)
    left = 1
    for site in range(n - 1):
        matrix = remainder.reshape(left * 2, -1)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        keep = min(bonds[site], len(s))
        tensors.append(u[:, :keep].reshape(left, 2, keep))
        remainder = (s[:keep, None] * vh[:keep]).reshape(keep, -1)
        left = keep
    tensors.append(remainder.reshape(left, 2, 1))
    return tensors


def random_init(bonds: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    dims = [1] + bonds + [1]
    return [
        rng.normal(size=(dims[i], 2, dims[i + 1]))
        + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))
        for i in range(len(dims) - 1)
    ]


def right_canonicalize(tensors: list[np.ndarray]) -> None:
    """Make tensors[1:] right-orthonormal in place (residual into site 0)."""
    for k in range(len(tensors) - 1, 0, -1):
        a, _, c = tensors[k].shape
        matrix = tensors[k].reshape(a, 2 * c)
        q, r = np.linalg.qr(matrix.conj().T)  # matrix^dag = q r
        keep = q.shape[1]
        tensors[k] = q.conj().T.reshape(keep, 2, c)
        tensors[k - 1] = np.einsum("xpa,ab->xpb", tensors[k - 1], r.conj().T)


def right_operators(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """RC[k]: contraction of conj(T[k..n-1]) alone, shape (D_k, 2^(n-k))."""
    n = len(tensors)
    rc = [None] * (n + 1)
    rc[n] = np.ones((1, 1), dtype=complex)
    for k in range(n - 1, -1, -1):
        # (a,p,b) x (b,t) -> (a, p*t)
        combined = np.einsum("apb,bt->apt", tensors[k].conj(), rc[k + 1])
        rc[k] = combined.reshape(tensors[k].shape[0], -1)
    return rc


def als_max_overlap(state: np.ndarray, bonds: list[int], init) -> float:
    """|<m|phi>| maximized over MPS(bonds): two-site DMRG-style sweeps.

    Two-site updates (optimal two-site block = normalized environment,
    then SVD truncation back to the bond profile) escape the fixed-rank
    traps that stall single-site ALS; the single-site scheme plateaued
    below the measured atom on instance 0, which the built-in
    diagnostic (delta* <= B_MPS) exposed.
    """
    n = len(bonds) + 1
    tensors = [t.astype(complex).copy() for t in init]
    overlap = 0.0
    for _ in range(SWEEPS):
        # The exact local update requires downstream tensors to be
        # right-orthonormal (isometry condition); upstream ones are
        # left-orthonormal isometries by the SVD split below.
        right_canonicalize(tensors)
        rc = right_operators(tensors)
        ell = state.reshape(1, -1).astype(complex)  # (D_k, 2^(n-k)) at k=0
        for k in range(n - 1):
            d_left = tensors[k].shape[0]
            d_right = tensors[k + 1].shape[2]
            tail = 2 ** (n - k - 2)
            ell4 = ell.reshape(d_left, 2, 2, tail)
            env2 = np.einsum("apqt,ct->apqc", ell4, rc[k + 2])
            matrix = env2.reshape(d_left * 2, 2 * d_right)
            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            keep = min(bonds[k], len(s))
            achieved = float(np.linalg.norm(s[:keep]))
            tensors[k] = u[:, :keep].reshape(d_left, 2, keep)
            block = (s[:keep, None] * vh[:keep]) / max(achieved, 1e-300)
            tensors[k + 1] = block.reshape(keep, 2, d_right)
            ell = np.einsum(
                "apb,apt->bt",
                tensors[k].conj(),
                ell.reshape(d_left, 2, 2 * tail),
            )
        previous, overlap = overlap, achieved
        if abs(overlap - previous) < TOL:
            break
    return overlap


def best_mps_overlap(
    state: np.ndarray, bonds: list[int], seed: int
) -> float:
    rng = np.random.default_rng(seed)
    best = als_max_overlap(state, bonds, truncation_init(state, bonds))
    for _ in range(RESTARTS - 1):
        best = max(
            best, als_max_overlap(state, bonds, random_init(bonds, rng))
        )
    return best**2


def self_tests() -> None:
    rng = np.random.default_rng(1)
    # (a) exact recovery of an MPS target at its own bond profile
    bonds = [2, 4, 4, 2, 1]  # n = 6, deliberately uneven
    target_tensors = random_init(bonds, rng)
    target = target_tensors[0]
    for t in target_tensors[1:]:
        target = np.einsum("...a,apb->...pb", target, t)
    target = target.reshape(-1)
    target /= np.linalg.norm(target)
    value = best_mps_overlap(target, bonds, seed=2)
    assert value > 1 - 1e-10, f"MPS self-recovery failed: {value}"
    # (b) Haar state at unrestricted bond -> overlap 1
    state = rng.normal(size=64) + 1j * rng.normal(size=64)
    state /= np.linalg.norm(state)
    full = [2, 4, 8, 4, 2]
    value = best_mps_overlap(state, full, seed=3)
    assert value > 1 - 1e-10, f"full-bond overlap not 1: {value}"
    # (c) monotonicity in bond dimension
    low = best_mps_overlap(state, [2, 2, 2, 2, 2], seed=4)
    high = best_mps_overlap(state, [2, 4, 4, 4, 2], seed=4)
    assert high >= low - 1e-9, f"monotonicity violated: {low} vs {high}"
    print(
        "self-tests passed (MPS recovery, full-bond exactness, "
        f"monotonicity {low:.4f} <= {high:.4f})"
    )


def main() -> None:
    # At tau_p = n/2 the corrected bond profile saturates every cut and
    # B_MPS = 1 (vacuous, see 09 Section 1). The informative regime is
    # shallow peaking; the confrontation below targets tau_p = 2 at
    # n = 8, matching results/shallow_peaking.
    self_tests()
    print(f"\n{'n':>4} {'tau_p':>6} {'inst':>4} {'bonds':>18} "
          f"{'B_MPS>=':>8} {'B_mincut':>9} {'measured':>9}")
    for instance in (0, 1, 2):
        config = CircuitConfig(8, 8, 2)
        state = scrambled_state(
            config,
            np.random.default_rng(
                np.random.SeedSequence(42, spawn_key=(8, instance))
            ),
        )
        bonds = bond_profile(config)
        value = best_mps_overlap(state, bonds, seed=100 + instance)
        mincut, _ = bound_for_state(state, config)
        assert value <= mincut + 1e-9, "ALS exceeded the min-cut bound"
        shallow = RESULTS / "shallow_peaking" / f"pq_n8_i{instance}_sigma0.1.npz"
        measured = (
            float(np.load(shallow)["peak_weights"].max())
            if shallow.exists()
            else float("nan")
        )
        bond_text = ",".join(str(b) for b in bonds)
        print(f"{8:>4} {2:>6} {instance:>4} {bond_text:>18} "
              f"{value:>8.4f} {mincut:>9.4f} {measured:>9.4f}")


if __name__ == "__main__":
    main()
