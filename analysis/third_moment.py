"""Exact third moment of the field via S3 spin transfer.

Same machinery as the k = 2 kernel (kernel_exact.py), lifted to three
copies: per-site labels live in S3, gates resolve misaligned labels with
Weingarten weights from the 6x6 Gram system at gate dimension d = 4, and
the diagonal boundary contracts each configuration against
X_c = <w^x3| sigma_hat(c) |w^x3> (per-site copy permutations of the probe).

Self-tests: gate idempotence on aligned labels, doubled-trace preservation,
Haar limit E[delta^3] -> 6/(D(D+1)(D+2)), and an independent Monte-Carlo
check of the full pipeline at n = 4 (bare numpy circuit simulator, random
probe vector).

Usage:
    python analysis/third_moment.py          # n = 6 exact, ~10 min
"""

from __future__ import annotations

import sys
import time
from itertools import permutations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401  (wires up the reproduction repo path)
from kernel_exact import probe_state_fn, spin_weights  # noqa: E402
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    brick_wall_pairs,
    sample_haar_random_layers,
)

PERMS = tuple(permutations(range(3)))  # the 6 elements of S3
IDENTITY = PERMS.index((0, 1, 2))
NUM_QUBITS = 6
PEAKING_LAYERS = 3
SNAPSHOT_DEPTHS = (1, 2, 3, 6, 12, 24)
BASE_SEED = 4041


def compose(p, q):
    """(p o q)(i) = p[q[i]]."""
    return tuple(p[q[i]] for i in range(3))


def inverse(p):
    out = [0, 0, 0]
    for i in range(3):
        out[p[i]] = i
    return tuple(out)


def num_cycles(p) -> int:
    seen, cycles = set(), 0
    for start in range(3):
        if start in seen:
            continue
        cycles += 1
        node = start
        while node not in seen:
            seen.add(node)
            node = p[node]
    return cycles


def transfer_tensor() -> np.ndarray:
    """T[a, b, s]: misaligned incoming (alpha, beta) -> aligned sigma."""
    gram = np.array(
        [
            [4.0 ** num_cycles(compose(sig, inverse(tau))) for tau in PERMS]
            for sig in PERMS
        ]
    )
    gram_inv = np.linalg.inv(gram)
    tensor = np.empty((6, 6, 6))
    for a, alpha in enumerate(PERMS):
        for b, beta in enumerate(PERMS):
            traces = np.array(
                [
                    2.0 ** num_cycles(compose(alpha, inverse(tau)))
                    * 2.0 ** num_cycles(compose(beta, inverse(tau)))
                    for tau in PERMS
                ]
            )
            tensor[a, b] = gram_inv @ traces
    return tensor


def s3_weights(num_qubits: int, depth: int, snapshots) -> dict[int, np.ndarray]:
    """Weights W_tau(c) over base-6 site configs, snapshotted at depths."""
    tensor = transfer_tensor()
    size = 6**num_qubits
    place = 6 ** np.arange(num_qubits)

    weights = np.zeros(size)
    first_pairs = brick_wall_pairs(0, num_qubits)
    for labels in product(range(6), repeat=len(first_pairs)):
        index = 0
        for label, (site_a, site_b) in zip(labels, first_pairs):
            index += label * (place[site_a] + place[site_b])
        weights[index] = (1.0 / 120.0) ** len(first_pairs)

    saved = {}
    if 1 in snapshots:
        saved[1] = weights.copy()
    indices = np.arange(size)
    for layer in range(1, depth):
        for site_a, site_b in brick_wall_pairs(layer, num_qubits):
            digit_a = (indices // place[site_a]) % 6
            digit_b = (indices // place[site_b]) % 6
            aligned = digit_a == digit_b
            updated = np.where(aligned, weights, 0.0)
            moving = ~aligned & (weights != 0.0)
            if np.any(moving):
                base = (
                    indices[moving]
                    - digit_a[moving] * place[site_a]
                    - digit_b[moving] * place[site_b]
                )
                coeff = tensor[digit_a[moving], digit_b[moving]]  # (m, 6)
                for label in range(6):
                    np.add.at(
                        updated,
                        base + label * (place[site_a] + place[site_b]),
                        weights[moving] * coeff[:, label],
                    )
            weights = updated
        if (layer + 1) in snapshots:
            saved[layer + 1] = weights.copy()
    return saved


def doubled_trace(weights: np.ndarray, num_qubits: int) -> float:
    """Trace of the averaged three-copy state; must be exactly 1."""
    indices = np.arange(len(weights))
    factors = np.array([2.0 ** num_cycles(p) for p in PERMS])
    total = np.ones(len(weights))
    place = 6 ** np.arange(num_qubits)
    for site in range(num_qubits):
        total *= factors[(indices // place[site]) % 6]
    return float(np.dot(weights, total))


def site_major_tensor(state: np.ndarray, num_qubits: int) -> np.ndarray:
    """|w>^x3 reshaped so each site carries its three copy bits (dim 8)."""
    triple = np.einsum("a,b,c->abc", state, state, state)
    triple = triple.reshape([2] * (3 * num_qubits))
    order = []
    for site in range(num_qubits):
        order += [site, num_qubits + site, 2 * num_qubits + site]
    return np.transpose(triple, order).reshape([8] * num_qubits)


def permutation_gathers() -> list[np.ndarray]:
    """Index map on the 8 local states for each sigma in S3.

    sigma_hat |b_0 b_1 b_2> = |b_{s(0)} b_{s(1)} b_{s(2)}> with s = sigma^-1.
    """
    maps = []
    for perm in PERMS:
        inv = inverse(perm)
        table = np.empty(8, dtype=np.int64)
        for bits in range(8):
            b = [(bits >> (2 - c)) & 1 for c in range(3)]
            permuted = [b[inv[c]] for c in range(3)]
            table[bits] = (permuted[0] << 2) | (permuted[1] << 1) | permuted[2]
        maps.append(table)
    return maps


def boundary_values(
    state: np.ndarray, num_qubits: int, configs: np.ndarray
) -> np.ndarray:
    """X_c = <w^x3| sigma_hat(c) |w^x3> for the requested configs."""
    tensor = site_major_tensor(state, num_qubits)
    flat = tensor.reshape(-1)
    gathers = permutation_gathers()
    place = 6 ** np.arange(num_qubits)
    values = np.empty(len(configs), dtype=complex)
    for row, config in enumerate(configs):
        current = tensor
        for site in range(num_qubits):
            label = int((config // place[site]) % 6)
            if label != IDENTITY:
                current = np.take(current, gathers[label], axis=site)
        values[row] = np.vdot(flat, current.reshape(-1))
    return values


def exact_third_moment(
    weights: np.ndarray, state: np.ndarray, num_qubits: int
) -> float:
    configs = np.flatnonzero(weights)
    values = boundary_values(state, num_qubits, configs)
    result = complex(np.dot(weights[configs], values))
    assert abs(result.imag) < 1e-10 * max(abs(result.real), 1e-300)
    return result.real


def exact_second_moment_k2(
    depth: int, state: np.ndarray, num_qubits: int
) -> float:
    from kernel_exact import exact_second_moment

    return exact_second_moment(
        spin_weights(num_qubits, depth), state, state, num_qubits
    )


def self_tests() -> None:
    tensor = transfer_tensor()
    for a in range(6):
        assert np.allclose(tensor[a, a], np.eye(6)[a]), "gate not idempotent"

    for depth in (1, 2, 5):
        saved = s3_weights(4, depth, (depth,))
        trace = doubled_trace(saved[depth], 4)
        assert abs(trace - 1.0) < 1e-10, f"trace {trace} at depth {depth}"

    dim = 2.0**4
    deep = s3_weights(4, 40, (40,))[40]
    rng = np.random.default_rng(5)
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)
    vector /= np.linalg.norm(vector)
    haar = 6.0 / (dim * (dim + 1.0) * (dim + 2.0))
    exact = exact_third_moment(deep, vector, 4)
    assert abs(exact - haar) / haar < 1e-8, f"Haar limit: {exact} vs {haar}"

    # Independent Monte-Carlo check of the full pipeline at n = 4, depth 2.
    def apply_gate(state, gate, site, n):
        tensor = state.reshape([2] * n)
        tensor = np.moveaxis(tensor, (site, site + 1), (0, 1))
        shape = tensor.shape
        tensor = (gate @ tensor.reshape(4, -1)).reshape(shape)
        return np.moveaxis(tensor, (0, 1), (site, site + 1)).reshape(-1)

    probe = rng.normal(size=16) + 1j * rng.normal(size=16)
    probe /= np.linalg.norm(probe)
    exact = exact_third_moment(s3_weights(4, 2, (2,))[2], probe, 4)
    samples = np.empty(60000)
    from scipy.stats import unitary_group

    mc_rng = np.random.default_rng(6)
    for k in range(len(samples)):
        state = np.zeros(16, dtype=complex)
        state[0] = 1.0
        for layer in range(2):
            for site, _ in brick_wall_pairs(layer, 4):
                state = apply_gate(
                    state, unitary_group.rvs(4, random_state=mc_rng), site, 4
                )
        samples[k] = np.abs(np.vdot(probe, state)) ** 6
    monte = samples.mean()
    stderr = samples.std() / np.sqrt(len(samples))
    assert abs(exact - monte) < 4 * stderr, (
        f"MC pipeline check: exact {exact:.3e} vs MC {monte:.3e} +/- {stderr:.1e}"
    )
    print(
        "self-tests passed (idempotence, trace, Haar limit, "
        f"MC pipeline: exact {exact:.4e} vs MC {monte:.4e} +/- {stderr:.1e})"
    )


def main() -> None:
    self_tests()
    dim = 2.0**NUM_QUBITS
    config = CircuitConfig(NUM_QUBITS, NUM_QUBITS, PEAKING_LAYERS)
    probe = probe_state_fn(config)
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED))

    start = time.perf_counter()
    saved = s3_weights(NUM_QUBITS, max(SNAPSHOT_DEPTHS), SNAPSHOT_DEPTHS)
    print(f"S3 transfer over {6**NUM_QUBITS} configs: {time.perf_counter()-start:.0f} s")
    for depth, weights in saved.items():
        trace = doubled_trace(weights, NUM_QUBITS)
        assert abs(trace - 1.0) < 1e-9, f"trace {trace} at depth {depth}"

    for scale in (0.1, 0.5):
        theta = rng.normal(0.0, scale, config.num_peaking_parameters)
        state = np.asarray(probe(theta))
        start = time.perf_counter()
        print(f"\nprobe init scale {scale:g}  (n = {NUM_QUBITS}, tau_p = {PEAKING_LAYERS})")
        print(f"{'tau_r':>6} {'m2':>8} {'m3':>8}   (Porter-Thomas: 1, 1)")
        for depth in SNAPSHOT_DEPTHS:
            third = exact_third_moment(saved[depth], state, NUM_QUBITS)
            second = exact_second_moment_k2(depth, state, NUM_QUBITS)
            m2 = second / (2.0 / dim**2)
            m3 = third / (6.0 / dim**3)
            print(f"{depth:>6} {m2:>8.3f} {m3:>8.3f}")
        print(f"[{time.perf_counter()-start:.0f} s]")


if __name__ == "__main__":
    main()
