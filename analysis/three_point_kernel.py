"""Off-diagonal three-point kernel of the field.

E[delta(t1) delta(t2) delta(t3)] = sum_c W3(c) <w1 x w2 x w3| sigma(c)
|w1 x w2 x w3> with the S3 spin weights of third_moment.py and a
distinct-probe boundary. The envelope three-point function is the exact
ratio to the Haar three-point formula, and the registered test compares
ln E[s1 s2 s3] with the Gaussian-log-field prediction
sum_{i<j} ln(1 + C_ij) built from the exact pairwise kernel.

Self-tests: diagonal case reproduces exact_third_moment; deep limit
reproduces the Haar permutation formula on random states; independent
Monte-Carlo pipeline at n = 4.

Usage:
    python analysis/three_point_kernel.py     # ~20 min, writes the Delta3 table
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from kernel_exact import exact_second_moment, probe_state_fn, spin_weights  # noqa: E402
from third_moment import (  # noqa: E402
    IDENTITY,
    PERMS,
    exact_third_moment,
    permutation_gathers,
    s3_weights,
)
from peaked_circuits import CircuitConfig  # noqa: E402

NUM_QUBITS = 6
DEPTH = 6
PEAKING_LAYERS = 3
PROBE_SCALE = 0.1
SHIFTS = (0.05, 0.1, 0.15, 0.2, 0.3)
BASE_SEED = 6067


def site_major_triple(states, num_qubits: int) -> np.ndarray:
    """|w1 x w2 x w3> reshaped so each site carries its three copy bits."""
    triple = np.einsum("a,b,c->abc", states[0], states[1], states[2])
    triple = triple.reshape([2] * (3 * num_qubits))
    order = []
    for site in range(num_qubits):
        order += [site, num_qubits + site, 2 * num_qubits + site]
    return np.transpose(triple, order).reshape([8] * num_qubits)


def boundary_values_triple(states, num_qubits: int, configs) -> np.ndarray:
    """X_c = <w1w2w3| sigma(c) |w1w2w3> for distinct probes."""
    tensor = site_major_triple(states, num_qubits)
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


def exact_three_point(weights, states, num_qubits: int) -> float:
    configs = np.flatnonzero(weights)
    values = boundary_values_triple(states, num_qubits, configs)
    result = complex(np.dot(weights[configs], values))
    assert abs(result.imag) < 1e-9 * max(abs(result.real), 1e-300)
    return result.real


def haar_three_point(states, num_qubits: int) -> float:
    """Deep limit: sum over S3 of cycle products / (D(D+1)(D+2))."""
    dim = 2.0**num_qubits
    overlaps = np.array(
        [[np.vdot(states[i], states[j]) for j in range(3)] for i in range(3)]
    )
    total = 0.0
    for perm in PERMS:
        term = 1.0 + 0.0j
        seen = set()
        for start in range(3):
            if start in seen:
                continue
            cycle = [start]
            node = perm[start]
            while node != start:
                cycle.append(node)
                node = perm[node]
            seen.update(cycle)
            factor = 1.0 + 0.0j
            for a, b in zip(cycle, cycle[1:] + cycle[:1]):
                factor *= overlaps[b][a]  # <w_b|w_a> along the cycle
            term *= factor
        total += term.real
    return total / (dim * (dim + 1.0) * (dim + 2.0))


def pairwise_envelope_cov(weights, state_a, state_b, num_qubits: int) -> float:
    dim = 2.0**num_qubits
    overlap = float(np.abs(np.vdot(state_a, state_b)) ** 2)
    exact = exact_second_moment(weights, state_a, state_b, num_qubits)
    return exact / ((1.0 + overlap) / (dim * (dim + 1.0))) - 1.0


def self_tests() -> None:
    rng = np.random.default_rng(9)
    # (i) diagonal case == exact_third_moment (n = 4, depth 2)
    weights4 = s3_weights(4, 2, (2,))[2]
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    diagonal = exact_three_point(weights4, [state, state, state], 4)
    reference = exact_third_moment(weights4, state, 4)
    assert abs(diagonal - reference) < 1e-12 * abs(reference), (diagonal, reference)

    # (ii) deep limit on three distinct random states (n = 4, depth 40)
    deep = s3_weights(4, 40, (40,))[40]
    states = []
    for _ in range(3):
        vec = rng.normal(size=16) + 1j * rng.normal(size=16)
        states.append(vec / np.linalg.norm(vec))
    exact = exact_three_point(deep, states, 4)
    haar = haar_three_point(states, 4)
    assert abs(exact - haar) / haar < 1e-8, (exact, haar)

    # (iii) independent Monte-Carlo pipeline (n = 4, depth 2)
    from ceiling_bound import scrambled_state

    config = CircuitConfig(4, 2, 2)
    samples = np.empty(40000)
    for k in range(len(samples)):
        phi = scrambled_state(
            config,
            np.random.default_rng(np.random.SeedSequence(707, spawn_key=(k,))),
        )
        d = [abs(np.vdot(s, phi)) ** 2 for s in states]
        samples[k] = d[0] * d[1] * d[2]
    exact = exact_three_point(s3_weights(4, 2, (2,))[2], states, 4)
    stderr = samples.std() / np.sqrt(len(samples))
    assert abs(exact - samples.mean()) < 4 * stderr, (
        exact, samples.mean(), stderr
    )
    print(
        "self-tests passed (diagonal, deep limit, MC pipeline: "
        f"exact {exact:.4e} vs MC {samples.mean():.4e} +/- {stderr:.1e})"
    )


def main() -> None:
    self_tests()
    config = CircuitConfig(NUM_QUBITS, DEPTH, PEAKING_LAYERS)
    probe = probe_state_fn(config)
    rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED))
    num_parameters = config.num_peaking_parameters
    theta = rng.normal(0.0, PROBE_SCALE, num_parameters)
    u = rng.normal(0.0, 1.0, num_parameters)
    u /= np.linalg.norm(u)
    v = rng.normal(0.0, 1.0, num_parameters)
    v -= u * (u @ v)
    v /= np.linalg.norm(v)

    start = time.perf_counter()
    weights3 = s3_weights(NUM_QUBITS, DEPTH, (DEPTH,))[DEPTH]
    weights2 = spin_weights(NUM_QUBITS, DEPTH)
    print(f"S3 weights ready [{time.perf_counter() - start:.0f} s]\n"
          f"{'geometry':>10} {'t':>5} {'E3/Haar3':>9} {'sum ln(1+Cij)':>13} "
          f"{'Delta3':>8}")
    for geometry in ("collinear", "triangle"):
        for t in SHIFTS:
            scale = t * np.sqrt(num_parameters)
            if geometry == "collinear":
                thetas = [theta, theta + scale * u, theta + 2 * scale * u]
            else:
                thetas = [theta, theta + scale * u, theta + scale * v]
            states = [np.asarray(probe(p)) for p in thetas]
            e3 = exact_three_point(weights3, states, NUM_QUBITS)
            h3 = haar_three_point(states, NUM_QUBITS)
            ratio = e3 / h3
            pair_sum = sum(
                np.log1p(pairwise_envelope_cov(weights2, states[i], states[j],
                                               NUM_QUBITS))
                for i in range(3) for j in range(i + 1, 3)
            )
            delta3 = float(np.log(ratio) - pair_sum)
            print(f"{geometry:>10} {t:>5.2f} {ratio:>9.4f} {pair_sum:>13.4f} "
                  f"{delta3:>+8.4f}")


if __name__ == "__main__":
    main()
