"""Exact fourth moment of the field via S4 spin transfer (04, section 8).

Same machinery as S3 (third_moment.py), lifted to four copies: per-site
labels in S4 (24 elements, linearly independent at gate dimension
d = 4 = k, the edge of validity), Weingarten gate rules from the 24x24
Gram system, init weight 1/840 (the S4 row sum at d = 4), sparse
dict-of-configs evolution (reachable configs stay gate-aligned, so the
support is bounded by 24^(gates per layer + free sites)).

Self-tests: gate idempotence, trace preservation (per-site factor
2^#cycles), Haar floor m4 -> D^3/((D+1)(D+2)(D+3)), independent
Monte-Carlo pipeline at n = 4.

Two config evaluators, cross-validated against each other at n = 4:
the dense gather (site-major 16^n tensor, fastest through n = 6) and a
per-config einsum contraction of the eight state copies (memory O(2^n)
per intermediate, the only feasible route at n = 8 where the dense
tensor would need 69 GB per worker).

Usage:
    python analysis/fourth_moment.py --self-tests-only
    python analysis/fourth_moment.py                    # n = 6 scan
    python analysis/fourth_moment.py --num-qubits 8 --budget 400000
"""

from __future__ import annotations

import argparse
import string
import sys
import time
from itertools import permutations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from kernel_exact import probe_state_fn, spin_weights  # noqa: E402
from kernel_exact import exact_second_moment  # noqa: E402
from peaked_circuits import CircuitConfig, brick_wall_pairs  # noqa: E402

K = 4
PERMS = tuple(permutations(range(K)))
IDENTITY = PERMS.index(tuple(range(K)))
INDEX = {p: i for i, p in enumerate(PERMS)}


def compose(p, q):
    return tuple(p[q[i]] for i in range(K))


def inverse(p):
    out = [0] * K
    for i in range(K):
        out[p[i]] = i
    return tuple(out)


def num_cycles(p) -> int:
    seen, cycles = set(), 0
    for start in range(K):
        if start in seen:
            continue
        cycles += 1
        node = start
        while node not in seen:
            seen.add(node)
            node = p[node]
    return cycles


def transfer_tensor() -> np.ndarray:
    gram = np.array(
        [[4.0 ** num_cycles(compose(s, inverse(t))) for t in PERMS]
         for s in PERMS]
    )
    gram_inv = np.linalg.inv(gram)
    tensor = np.empty((len(PERMS), len(PERMS), len(PERMS)))
    for a, alpha in enumerate(PERMS):
        for b, beta in enumerate(PERMS):
            traces = np.array(
                [2.0 ** num_cycles(compose(alpha, inverse(t)))
                 * 2.0 ** num_cycles(compose(beta, inverse(t)))
                 for t in PERMS]
            )
            tensor[a, b] = gram_inv @ traces
    return tensor


def s4_weights(num_qubits: int, depth: int) -> dict[tuple, float]:
    """Sparse weights over site-label configs (tuples of S4 indices)."""
    tensor = transfer_tensor()
    init_weight = 1.0 / np.array(
        [[4.0 ** num_cycles(compose(s, inverse(t))) for t in PERMS]
         for s in PERMS]
    ).sum(axis=1)[0]  # row sum = 840 at d = 4

    first_pairs = brick_wall_pairs(0, num_qubits)
    weights: dict[tuple, float] = {}
    for labels in product(range(len(PERMS)), repeat=len(first_pairs)):
        config = [IDENTITY] * num_qubits
        for label, (site_a, site_b) in zip(labels, first_pairs):
            config[site_a] = label
            config[site_b] = label
        weights[tuple(config)] = init_weight ** len(first_pairs)

    for layer in range(1, depth):
        for site_a, site_b in brick_wall_pairs(layer, num_qubits):
            updated: dict[tuple, float] = {}
            for config, weight in weights.items():
                a, b = config[site_a], config[site_b]
                if a == b:
                    updated[config] = updated.get(config, 0.0) + weight
                else:
                    coeffs = tensor[a, b]
                    base = list(config)
                    for label in range(len(PERMS)):
                        coefficient = coeffs[label]
                        if coefficient == 0.0:
                            continue
                        base[site_a] = label
                        base[site_b] = label
                        key = tuple(base)
                        updated[key] = updated.get(key, 0.0) \
                            + weight * coefficient
            weights = updated
    return weights


def doubled_trace(weights: dict[tuple, float], num_qubits: int) -> float:
    factors = np.array([2.0 ** num_cycles(p) for p in PERMS])
    total = 0.0
    for config, weight in weights.items():
        term = weight
        for label in config:
            term *= factors[label]
        total += term
    return float(total)


def permutation_gathers() -> list[np.ndarray]:
    """Index maps on the 16 local basis states for each sigma in S4."""
    maps = []
    for perm in PERMS:
        inv = inverse(perm)
        table = np.empty(2**K, dtype=np.int64)
        for bits in range(2**K):
            b = [(bits >> (K - 1 - c)) & 1 for c in range(K)]
            permuted = [b[inv[c]] for c in range(K)]
            value = 0
            for bit in permuted:
                value = (value << 1) | bit
            table[bits] = value
        maps.append(table)
    return maps


def site_major(state: np.ndarray, num_qubits: int) -> np.ndarray:
    tensor = np.einsum("a,b,c,d->abcd", state, state, state, state)
    tensor = tensor.reshape([2] * (K * num_qubits))
    order = []
    for site in range(num_qubits):
        order += [copy * num_qubits + site for copy in range(K)]
    return np.transpose(tensor, order).reshape([2**K] * num_qubits)


_WORKER_TENSOR = None


def _chunk_worker(payload):
    """Evaluate sum_c W(c) X_c over one chunk of configs (worker side)."""
    state, num_qubits, chunk = payload
    global _WORKER_TENSOR
    if _WORKER_TENSOR is None or _WORKER_TENSOR[0] is not state:
        tensor = site_major(np.asarray(state), num_qubits)
        _WORKER_TENSOR = (state, tensor, tensor.reshape(-1),
                          permutation_gathers())
    _, tensor, flat, gathers = _WORKER_TENSOR
    total = 0.0 + 0.0j
    for config, weight in chunk:
        current = tensor
        for site, label in enumerate(config):
            if label != IDENTITY:
                current = np.take(current, gathers[label], axis=site)
        total += weight * np.vdot(flat, current.reshape(-1))
    return total


def _einsum_subscripts(config) -> str:
    """Subscripts for X_c as one contraction of 4 bra + 4 ket copies.

    Index letter (k, s) is bit s of copy k; ket copy k reads, at site s,
    the bit of bra copy inv(pi_s)(k) — the same wiring the dense gather
    tables implement (see permutation_gathers)."""
    n = len(config)
    letters = string.ascii_letters

    def letter(k: int, s: int) -> str:
        return letters[k * n + s]

    bras = ["".join(letter(k, s) for s in range(n)) for k in range(K)]
    kets = []
    for k in range(K):
        inv_tables = [inverse(PERMS[label]) for label in config]
        kets.append(
            "".join(letter(inv_tables[s][k], s) for s in range(n))
        )
    return ",".join(bras + kets) + "->"


def _einsum_chunk_worker(payload):
    """Evaluate sum_c W(c) X_c over one chunk, one einsum per config."""
    state, num_qubits, chunk = payload
    psi = np.asarray(state).reshape([2] * num_qubits)
    operands = [np.conj(psi)] * K + [psi] * K
    paths: dict[str, list] = {}
    total = 0.0 + 0.0j
    for config, weight in chunk:
        sub = _einsum_subscripts(config)
        path = paths.get(sub)
        if path is None:
            # The default memory limit (= largest input) forbids every
            # pairwise step here and collapses to the naive 2^(4n) sum;
            # raising it lets greedy find near-optimal paths (~1e5 FLOPs
            # per config at n = 8, intermediates ~4e3 elements).
            path = np.einsum_path(
                sub, *operands, optimize=("greedy", 2**24)
            )[0]
            paths[sub] = path
        total += weight * np.einsum(sub, *operands, optimize=path)
    return total


_CHUNK_WORKERS = {"dense": _chunk_worker, "einsum": _einsum_chunk_worker}


def exact_fourth_moment(
    weights: dict[tuple, float], state: np.ndarray, num_qubits: int,
    budget: int | None = None, workers: int = 1,
    evaluator: str = "dense",
) -> tuple[float, float]:
    """Sum_c W(c) X_c with certified truncation, chunk-parallel."""
    worker_fn = _CHUNK_WORKERS[evaluator]
    items = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
    dropped = 0.0
    if budget is not None and len(items) > budget:
        dropped = float(sum(abs(v) for _, v in items[budget:]))
        items = items[:budget]
    if workers <= 1 or len(items) < 2000:
        total = worker_fn((state, num_qubits, items))
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        chunks = [items[k::workers] for k in range(workers)]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            total = sum(
                pool.map(worker_fn,
                         [(state, num_qubits, chunk) for chunk in chunks])
            )
    assert abs(total.imag) < 1e-9 * max(abs(total.real), 1e-300)
    return float(total.real), dropped


def self_tests() -> None:
    tensor = transfer_tensor()
    for a in range(len(PERMS)):
        assert np.allclose(tensor[a, a], np.eye(len(PERMS))[a]), "idempotence"

    for depth in (1, 2, 4):
        weights = s4_weights(4, depth)
        trace = doubled_trace(weights, 4)
        assert abs(trace - 1.0) < 1e-9, f"trace {trace} at depth {depth}"

    dim = 2.0**4
    deep = s4_weights(4, 30)
    rng = np.random.default_rng(41)
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)
    vector /= np.linalg.norm(vector)
    haar = 24.0 / (dim * (dim + 1) * (dim + 2) * (dim + 3))
    exact, _ = exact_fourth_moment(deep, vector, 4)
    assert abs(exact - haar) / haar < 1e-8, (exact, haar)

    # Einsum evaluator vs dense gather: identical config values (50
    # heaviest configs, one by one) and identical full row at n = 4.
    shallow = s4_weights(4, 2)
    heavy = sorted(shallow.items(), key=lambda kv: -abs(kv[1]))[:50]
    for config, weight in heavy:
        dense_value = _chunk_worker((vector, 4, [(config, 1.0)]))
        einsum_value = _einsum_chunk_worker((vector, 4, [(config, 1.0)]))
        assert abs(dense_value - einsum_value) < 1e-11, (
            config, dense_value, einsum_value
        )
    dense_row, _ = exact_fourth_moment(shallow, vector, 4)
    einsum_row, _ = exact_fourth_moment(
        shallow, vector, 4, evaluator="einsum"
    )
    assert abs(dense_row - einsum_row) < 1e-10 * abs(dense_row), (
        dense_row, einsum_row
    )

    # Independent Monte-Carlo pipeline at n = 4, depth 2.
    from ceiling_bound import scrambled_state

    config = CircuitConfig(4, 2, 2)
    exact, _ = exact_fourth_moment(s4_weights(4, 2), vector, 4)
    samples = np.empty(60000)
    for k in range(len(samples)):
        phi = scrambled_state(
            config,
            np.random.default_rng(np.random.SeedSequence(808, spawn_key=(k,))),
        )
        samples[k] = abs(np.vdot(vector, phi)) ** 8
    stderr = samples.std() / np.sqrt(len(samples))
    assert abs(exact - samples.mean()) < 4 * stderr, (
        exact, samples.mean(), stderr
    )
    print("self-tests passed (idempotence, trace, Haar floor, MC pipeline: "
          f"exact {exact:.4e} vs MC {samples.mean():.4e} +/- {stderr:.1e})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-tests-only", action="store_true")
    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--budget", type=int, default=150000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluator", choices=("auto", "dense", "einsum"),
                        default="auto")
    args = parser.parse_args()

    self_tests()
    if args.self_tests_only:
        return

    num_qubits = args.num_qubits
    peaking = num_qubits // 2
    evaluator = args.evaluator
    if evaluator == "auto":
        evaluator = "dense" if num_qubits <= 6 else "einsum"
    dim = 2.0**num_qubits
    config = CircuitConfig(num_qubits, num_qubits, peaking)
    probe = probe_state_fn(config)
    rng = np.random.default_rng(np.random.SeedSequence(4042))
    theta = rng.normal(0.0, 0.1, config.num_peaking_parameters)
    state = np.asarray(probe(theta))

    print(f"\nn = {num_qubits}, tau_p = {peaking}, evaluator = {evaluator}")
    print(f"{'tau_r':>6} {'m2':>8} {'m3*':>8} {'m4':>8} {'ln m4/ln m2':>12} "
          f"{'dropped':>9}   (* m3 from third_moment for context)")
    from third_moment import exact_third_moment, s3_weights

    for depth in (1, 2, 3, num_qubits, 2 * num_qubits):
        start = time.perf_counter()
        weights4 = s4_weights(num_qubits, depth)
        fourth, dropped = exact_fourth_moment(
            weights4, state, num_qubits, budget=args.budget,
            workers=args.workers, evaluator=evaluator,
        )
        second = exact_second_moment(
            spin_weights(num_qubits, depth), state, state, num_qubits
        )
        weights3 = s3_weights(num_qubits, depth, (depth,))[depth]
        if num_qubits <= 6:
            third = exact_third_moment(weights3, state, num_qubits)
            third_bound = 0.0
        else:
            from third_moment_n8 import truncated_third

            third, third_bound = truncated_third(
                weights3, state, num_qubits, budget=6000
            )
        m2 = second / (2.0 / dim**2)
        m3 = third / (6.0 / dim**3)
        m4 = fourth / (24.0 / dim**4)
        exponent = np.log(m4) / np.log(m2) if m2 > 1 else float("nan")
        print(f"{depth:>6} {m2:>8.3f} {m3:>8.3f} {m4:>8.3f} {exponent:>12.2f} "
              f"{dropped:>9.1e} [{time.perf_counter() - start:.0f} s, "
              f"{len(weights4)} configs]")
        if dropped > 0.0:
            print(f"       certified |m4 error| <= {dropped / (24.0 / dim**4):.3e} "
                  f"(truncation, m4 units)")
        if third_bound > 0.0:
            print(f"       certified |m3 error| <= {third_bound / (6.0 / dim**3):.3e} "
                  f"(truncation, m3 units)")


if __name__ == "__main__":
    main()
