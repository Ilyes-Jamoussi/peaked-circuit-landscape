"""Exact gradient statistics of the field at initialization.

Cov[d_a delta, d_b delta] = sum_c W_tau(c) Tr[d_a rho_R(c) d_b rho_R(c)],
with the spin-transfer weights of kernel_exact and a differentiated
boundary. Derivative states are exact: the code's gate is the ordered
product of Pauli rotations exp(-i theta_k P_k / 2) (qml.ArbitraryUnitary
decomposition, verified against pennylane below), so d_k of a gate is an
operator insertion of (-i/2) P_k inside the product -- no matrix
calculus needed.

Self-tests (all must pass before any table is printed):
  (a) numpy V(theta)|0> matches the pennylane probe state bit-for-bit;
  (b) derivative states match 4th-order finite differences of w;
  (c) the exact trace covariance converges to the differentiated Haar
      kernel at deep tau_r;
  (d) independent Monte-Carlo: finite-difference gradients of delta over
      fresh instances at n = 6 agree with the exact formula (bootstrap).

Usage:
    python analysis/init_statistics.py          # scan + self-tests, ~20 min
    python analysis/init_statistics.py --self-tests-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from kernel_exact import probe_state_fn, spin_weights  # noqa: E402
from peaked_circuits import CircuitConfig, brick_wall_pairs  # noqa: E402

#: Pauli word order of qml.ArbitraryUnitary's decomposition
#: (pennylane _all_pauli_words_but_identity(2), extracted 2026-07-17).
WORDS = ["XI", "YI", "ZI", "ZX", "IX", "XX", "YX", "YY", "ZY", "IY",
         "XY", "XZ", "YZ", "ZZ", "IZ"]
_P1 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}
PAULIS = [np.kron(_P1[w[0]], _P1[w[1]]) for w in WORDS]


def rotation(word_index: int, angle: float) -> np.ndarray:
    p = PAULIS[word_index]
    return np.cos(angle / 2) * np.eye(4) - 1j * np.sin(angle / 2) * p


def gate_matrix(angles: np.ndarray, derivative: int | None = None) -> np.ndarray:
    """Ordered product of the 15 Pauli rotations; optional d/d angle_k.

    The product order (first rotation applied first to the state) is
    fixed by self-test (a) against pennylane.
    """
    matrix = np.eye(4, dtype=complex)
    for k in range(15):
        rot = rotation(k, angles[k])
        if derivative == k:
            rot = (-0.5j) * PAULIS[k] @ rot
        matrix = rot @ matrix
    return matrix


def apply_two_qubit(state, gate, site, n):
    tensor = state.reshape([2] * n)
    tensor = np.moveaxis(tensor, (site, site + 1), (0, 1))
    shape = tensor.shape
    tensor = (gate @ tensor.reshape(4, -1)).reshape(shape)
    return np.moveaxis(tensor, (0, 1), (site, site + 1)).reshape(-1)


def peaking_gates(config: CircuitConfig):
    """(layer_index, site) per gate, in application order of V."""
    gates = []
    for layer in range(config.num_peaking_layers):
        layer_index = config.num_random_layers + layer
        for site, _ in brick_wall_pairs(layer_index, config.num_qubits):
            gates.append(site)
    return gates


def probe_and_derivatives(config: CircuitConfig, theta: np.ndarray):
    """w = V(theta)^dag |0> and all P derivative states d_a w, exactly.

    V applies its gates in order; V^dag applies daggered gates in
    reverse order. d_a V^dag inserts the daggered derivative at gate g.
    """
    n = config.num_qubits
    sites = peaking_gates(config)
    per_gate = theta.reshape(len(sites), 15)
    dim = 2**n

    gate_mats = [gate_matrix(per_gate[g]) for g in range(len(sites))]
    zero = np.zeros(dim, dtype=complex)
    zero[0] = 1.0

    # Suffix states: s[j] = (gates j.. applied as daggers in reverse)|0>
    # built from the top: V^dag = G_0^dag G_1^dag ... applied to |0> in
    # reverse gate order.
    suffix = [zero]
    for g in range(len(sites) - 1, -1, -1):
        suffix.append(apply_two_qubit(suffix[-1], gate_mats[g].conj().T,
                                      sites[g], n))
    suffix = suffix[::-1]  # suffix[g] = (G_g^dag ... G_last^dag)|0>
    w = suffix[0]

    derivatives = np.empty((len(sites) * 15, dim), dtype=complex)
    for g in range(len(sites)):
        # prefix operator applied AFTER gate g's dagger:
        # d_(g,k) w = G_0^dag ... G_{g-1}^dag (d_k G_g)^dag suffix[g+1]
        for k in range(15):
            dgate = gate_matrix(per_gate[g], derivative=k)
            vec = apply_two_qubit(suffix[g + 1], dgate.conj().T, sites[g], n)
            for gg in range(g - 1, -1, -1):
                vec = apply_two_qubit(vec, gate_mats[gg].conj().T,
                                      sites[gg], n)
            derivatives[g * 15 + k] = vec
    return w, derivatives


def trace_gradient_covariance(
    weights: np.ndarray, w: np.ndarray, derivatives: np.ndarray, n: int
) -> float:
    """Sum_a Var[d_a delta] = sum_c W(c) sum_a Tr[(d_a rho_R)^2], exact.

    Tr[(A + A^dag)^2] = 2 Re Tr[A^2] + 2 Tr[A A^dag] with
    A = Tr_(bar R)|u><w|; vectorized over all derivative states.
    """
    num_derivatives = len(derivatives)
    total = 0.0
    for config_bits in np.flatnonzero(weights):
        region = [q for q in range(n) if config_bits & (1 << q)]
        complement = [q for q in range(n) if not config_bits & (1 << q)]
        r = len(region)
        w_mat = np.transpose(w.reshape([2] * n), region + complement).reshape(
            2**r, -1
        )
        u_stack = np.transpose(
            derivatives.reshape([num_derivatives] + [2] * n),
            [0] + [q + 1 for q in region] + [q + 1 for q in complement],
        ).reshape(num_derivatives, 2**r, -1)
        if r <= n - r:
            a_stack = u_stack @ w_mat.conj().T  # (P, 2^r, 2^r)
            cross = np.einsum("pij,pji->p", a_stack, a_stack).real
            gram = np.einsum("pij,pij->p", a_stack, a_stack.conj()).real
        else:
            # Work in the complement dimension (trace cyclicity):
            # Tr[(U W^+)^2] = Tr[(W^+ U)^2] and
            # Tr[U W^+ W U^+] = Tr[(W^+ W)(U^+ U)] — the (P, 2^r, 2^r)
            # stack at r = n would need 145 GB at n = 12.
            b_stack = w_mat.conj().T @ u_stack  # (P, 2^c, 2^c)
            cross = np.einsum("pij,pji->p", b_stack, b_stack).real
            w_gram = w_mat.conj().T @ w_mat  # (2^c, 2^c)
            u_gram = np.einsum(
                "pri,prj->pij", u_stack.conj(), u_stack
            )  # (P, 2^c, 2^c)
            gram = np.einsum("ij,pji->p", w_gram, u_gram).real
        total += weights[config_bits] * float(2.0 * (cross + gram).sum())
    return total


def haar_trace_covariance(w: np.ndarray, derivatives: np.ndarray, n: int) -> float:
    """Deep-limit value from differentiating (1+F)/(D(D+1)).

    d_a d'_a F at coincidence = 2<u|u> + <u|w>^2 + <w|u>^2 with u = d_a w;
    since <u|w> is purely imaginary (norm preservation), this equals
    2(<u|u> - |<u|w>|^2) = 2 x the Fubini-Study metric g_aa. Hence
    Tr Cov_Haar[grad] = 2 sum_a g_aa / (D(D+1)).
    """
    dim = 2.0**n
    total = 0.0
    for u in derivatives:
        ovl = np.vdot(w, u)
        total += np.vdot(u, u).real - (ovl * ovl.conjugate()).real
    return 2.0 * total / (dim * (dim + 1.0))


def self_tests() -> None:
    rng = np.random.default_rng(11)
    config = CircuitConfig(6, 6, 3)
    theta = rng.normal(0.0, 0.3, config.num_peaking_parameters)

    # (a) numpy probe vs pennylane probe
    w, derivatives = probe_and_derivatives(config, theta)
    w_pl = np.asarray(probe_state_fn(config)(theta))
    assert np.allclose(w, w_pl, atol=1e-10), (
        f"probe mismatch: {np.abs(w - w_pl).max():.2e}"
    )

    # (b) derivatives vs 4th-order central finite differences
    step = 1e-3
    for a in (0, 7, 25, len(theta) - 1):
        stencil = []
        for shift in (-2, -1, 1, 2):
            shifted = theta.copy()
            shifted[a] += shift * step
            stencil.append(probe_and_derivatives(config, shifted)[0])
        numeric = (stencil[0] - 8 * stencil[1] + 8 * stencil[2] - stencil[3]) / (
            12 * step
        )
        assert np.allclose(numeric, derivatives[a], atol=1e-8), (
            f"derivative {a} mismatch: "
            f"{np.abs(numeric - derivatives[a]).max():.2e}"
        )

    # (c) deep limit
    deep = spin_weights(6, 40)
    exact = trace_gradient_covariance(deep, w, derivatives, 6)
    haar = haar_trace_covariance(w, derivatives, 6)
    assert abs(exact - haar) / haar < 1e-6, f"deep limit: {exact} vs {haar}"

    # (d) small-dimension branch (trace cyclicity) vs the explicit
    # per-derivative A = U W^+ reference, all region sizes at depth 2.
    weights2 = spin_weights(6, 2)
    fast = trace_gradient_covariance(weights2, w, derivatives, 6)
    reference = 0.0
    for config_bits in np.flatnonzero(weights2):
        region = [q for q in range(6) if config_bits & (1 << q)]
        complement = [q for q in range(6) if not config_bits & (1 << q)]
        w_mat = np.transpose(
            w.reshape([2] * 6), region + complement
        ).reshape(2 ** len(region), -1)
        contribution = 0.0
        for u in derivatives:
            u_mat = np.transpose(
                u.reshape([2] * 6), region + complement
            ).reshape(2 ** len(region), -1)
            a = u_mat @ w_mat.conj().T
            contribution += 2.0 * (
                np.trace(a @ a).real + np.trace(a @ a.conj().T).real
            )
        reference += weights2[config_bits] * contribution
    assert abs(fast - reference) < 1e-9 * max(abs(reference), 1e-300), (
        fast, reference
    )

    print("self-tests (a)-(d) passed (probe match, 4 derivatives, "
          "deep limit, cyclic branch)")


def monte_carlo_check(num_instances: int = 200) -> None:
    """(d) independent pipeline: finite-difference gradients of delta over
    fresh Haar instances, against the exact formula. n = 6."""
    from ceiling_bound import scrambled_state

    rng = np.random.default_rng(17)
    config = CircuitConfig(6, 3, 3)
    theta = rng.normal(0.0, 0.1, config.num_peaking_parameters)
    w, derivatives = probe_and_derivatives(config, theta)
    weights = spin_weights(6, 3)
    exact = trace_gradient_covariance(weights, w, derivatives, 6)

    step = 1e-4
    totals = np.empty(num_instances)
    for i in range(num_instances):
        phi = scrambled_state(
            config, np.random.default_rng(np.random.SeedSequence(909, spawn_key=(i,)))
        )
        grad_sq = 0.0
        for a in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[a] += step
            minus[a] -= step
            wp = probe_and_derivatives_cheap(config, plus)
            wm = probe_and_derivatives_cheap(config, minus)
            d = (abs(np.vdot(wp, phi)) ** 2 - abs(np.vdot(wm, phi)) ** 2) / (
                2 * step
            )
            grad_sq += d * d
        totals[i] = grad_sq
    boot = np.random.default_rng(1)
    resampled = [
        totals[boot.integers(num_instances, size=num_instances)].mean()
        for _ in range(500)
    ]
    stderr = float(np.std(resampled))
    verdict = "ok" if abs(totals.mean() - exact) < 3 * stderr else "MISMATCH"
    print(
        f"MC check (n=6, tau_r=3, {num_instances} instances): "
        f"exact {exact:.4e} vs MC {totals.mean():.4e} +/- {stderr:.1e}  {verdict}"
    )
    assert verdict == "ok"


def probe_and_derivatives_cheap(config, theta):
    """w only (no derivatives) — for the finite-difference MC."""
    n = config.num_qubits
    sites = peaking_gates(config)
    per_gate = theta.reshape(len(sites), 15)
    zero = np.zeros(2**n, dtype=complex)
    zero[0] = 1.0
    vec = zero
    for g in range(len(sites) - 1, -1, -1):
        vec = apply_two_qubit(vec, gate_matrix(per_gate[g]).conj().T,
                              sites[g], n)
    return vec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-tests-only", action="store_true")
    parser.add_argument("--max-qubits", type=int, default=12)
    args = parser.parse_args()

    self_tests()
    monte_carlo_check()
    if args.self_tests_only:
        return

    print(f"\n{'n':>4} {'tau_r':>6} {'theta':>8} {'Tr Cov[grad]':>13} "
          f"{'Haar limit':>11} {'ratio':>7}")
    rng = np.random.default_rng(2026)
    for n in (6, 8, 10, 12):
        if n > args.max_qubits:
            break
        depths = sorted({1, 2, n // 2, n, 2 * n, 4 * n})
        config = CircuitConfig(n, n, n // 2)
        points = [("zero", np.zeros(config.num_peaking_parameters))]
        points += [
            (f"0.1#{k}", rng.normal(0.0, 0.1, config.num_peaking_parameters))
            for k in range(2)
        ]
        for label, theta in points:
            start = time.perf_counter()
            w, derivatives = probe_and_derivatives(config, theta)
            haar = haar_trace_covariance(w, derivatives, n)
            for depth in depths:
                weights = spin_weights(n, depth)
                exact = trace_gradient_covariance(weights, w, derivatives, n)
                print(f"{n:>4} {depth:>6} {label:>8} {exact:>13.4e} "
                      f"{haar:>11.4e} {exact / haar:>7.3f}")
            print(f"    [{time.perf_counter() - start:.0f} s]")


if __name__ == "__main__":
    main()
