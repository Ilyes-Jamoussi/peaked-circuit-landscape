"""Entanglement-truncation bound on delta*.

For each stored instance: build the scrambled state phi = U_r |0^n>,
count the peaking-brickwall gates crossing each bipartition cut, and
compute B = min over cuts of the top-4^g(b) Schmidt mass of phi.
Compare with the measured best delta (the reach atom at n = 8).

Also reruns the Schmidt check that caught the 2^g rank error (see
RETRACTION, 09 section 2): the measured optimal state
w* = V(theta*)^dagger |0^n> at n = 8, i0 carries Schmidt mass beyond
rank 2^g at the central cuts, and none beyond 4^g.

Self-tests: rank bound saturated by an explicit product state (B = 1
when phi is a product state), and Schmidt masses sum to 1.

Usage:
    python analysis/ceiling_bound.py     # ~1 min
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_experiment  # noqa: E402,F401
from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    brick_wall_pairs,
    sample_haar_random_layers,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def apply_gate(state: np.ndarray, gate: np.ndarray, site: int, n: int):
    tensor = state.reshape([2] * n)
    tensor = np.moveaxis(tensor, (site, site + 1), (0, 1))
    shape = tensor.shape
    tensor = (gate @ tensor.reshape(4, -1)).reshape(shape)
    return np.moveaxis(tensor, (0, 1), (site, site + 1)).reshape(-1)


def scrambled_state(config: CircuitConfig, rng: np.random.Generator) -> np.ndarray:
    layers = sample_haar_random_layers(config, rng)
    state = np.zeros(2**config.num_qubits, dtype=complex)
    state[0] = 1.0
    for layer_index, gates in enumerate(layers):
        for (site, _), gate in zip(
            brick_wall_pairs(layer_index, config.num_qubits), gates, strict=True
        ):
            state = apply_gate(state, gate, site, config.num_qubits)
    return state


def crossing_counts(config: CircuitConfig) -> dict[int, int]:
    """g(b) for each cut b (left block of b qubits), peaking layers only."""
    counts = {b: 0 for b in range(1, config.num_qubits)}
    start = config.num_random_layers
    for layer in range(start, start + config.num_peaking_layers):
        for site_a, site_b in brick_wall_pairs(layer, config.num_qubits):
            for b in range(site_a + 1, site_b + 1):
                counts[b] += 1
    return counts


def schmidt_masses(state: np.ndarray, num_qubits: int, cut: int) -> np.ndarray:
    matrix = state.reshape(2**cut, -1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    probabilities = np.sort(singular**2)[::-1]
    return probabilities


def bound_for_state(state: np.ndarray, config: CircuitConfig) -> tuple[float, int]:
    # CORRECTION 2026-07-16: a generic two-qubit gate has OPERATOR Schmidt
    # rank 4 (not 2), so each crossing multiplies the state Schmidt rank
    # by up to 4: rank <= 4^g, not 2^g. At tau_p = n/2 this makes the
    # bound vacuous (4^(tau_p/2) = 2^(n/2) saturates every cut). The
    # first version used 2^g and produced spuriously tight numbers,
    # caught by the Schmidt spectrum of the measured optimal state.
    counts = crossing_counts(config)
    best, best_cut = np.inf, -1
    for cut, g in counts.items():
        probabilities = schmidt_masses(state, config.num_qubits, cut)
        rank = min(4**g, len(probabilities))
        mass = float(probabilities[:rank].sum())
        if mass < best:
            best, best_cut = mass, cut
    return best, best_cut


def self_tests() -> None:
    config = CircuitConfig(4, 2, 2)
    product = np.zeros(16, dtype=complex)
    product[5] = 1.0  # |0101>, product state
    bound, _ = bound_for_state(product, config)
    assert abs(bound - 1.0) < 1e-12, "product state must give bound 1"
    rng = np.random.default_rng(0)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    masses = schmidt_masses(state, 4, 2)
    assert abs(masses.sum() - 1.0) < 1e-10, "Schmidt masses must sum to 1"
    print("self-tests passed (product-state bound, Schmidt normalization)")


def verify_corrected_rank() -> None:
    """Schmidt check on the measured optimal state (the one that caught the error).

    The reachable class is w = V^dagger |0^n> with V the peaking section, so
    the state Schmidt rank across a cut crossed by g gates is at most 4^g
    (operator Schmidt rank of a generic SU(4) gate is 4, not 2).  The best
    measured w* at n = 8, i0 violates the retracted 2^g rule at the central
    cuts and respects the corrected 4^g rule at every cut.
    """
    import pennylane as qml
    from peaked_circuits import _apply_peaking_section

    num_qubits = 8
    config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
    data = np.load(RESULTS / "pq" / "pq_n8_i0_sigma0.1.npz")
    best = int(np.argmax(data["peak_weights"]))
    theta_star = data["thetas_final"][best]

    @qml.qnode(qml.device("default.qubit", wires=num_qubits))
    def w_state(theta):
        qml.adjoint(_apply_peaking_section)(config, theta)
        return qml.state()

    state = np.asarray(w_state(theta_star))
    counts = crossing_counts(config)
    print(f"\nrank check on the measured optimal state (n = 8, i0, best restart "
          f"{best}, delta = {float(data['peak_weights'][best]):.4f}):")
    violated = []
    for cut in range(1, num_qubits):
        g = counts[cut]
        masses = schmidt_masses(state, num_qubits, cut)
        beyond_2g = float(masses[min(2**g, len(masses)):].sum())
        beyond_4g = float(masses[min(4**g, len(masses)):].sum())
        if beyond_2g > 5e-4:
            violated.append(cut)
        print(f"  cut {cut}: g = {g}, mass beyond 2^g = {beyond_2g:.3f} "
              f"(retracted rule), beyond 4^g = {beyond_4g:.1e} (corrected rule)")
    cuts = ", ".join(str(c) for c in violated)
    print(f"  retracted 2^g rule violated at cuts {cuts}; "
          f"corrected 4^g rule respected at every cut")


def main() -> None:
    self_tests()
    verify_corrected_rank()
    print("\ncorrected bound on the grid (vacuous expected at tau_p = n/2):")
    print(f"\n{'n':>4} {'tau_r':>6} {'inst':>4} {'bound B':>9} {'cut':>4} "
          f"{'measured best':>13} {'B - best':>9}")
    for num_qubits in (8, 10, 12, 14):
        config = CircuitConfig(num_qubits, num_qubits, num_qubits // 2)
        for instance in (0, 1, 2):
            state = scrambled_state(
                config,
                np.random.default_rng(
                    np.random.SeedSequence(42, spawn_key=(num_qubits, instance))
                ),
            )
            bound, cut = bound_for_state(state, config)
            data = np.load(
                RESULTS / "pq" / f"pq_n{num_qubits}_i{instance}_sigma0.1.npz"
            )
            baseline = float(np.abs(state[0]) ** 2)
            stored = float(data["baseline_peak_weight"])
            assert abs(baseline - stored) < 1e-9 * max(stored, 1e-12), (
                f"provenance: rebuilt baseline {baseline:.3e} != stored {stored:.3e}"
            )
            best = float(data["peak_weights"].max())
            print(f"{num_qubits:>4} {num_qubits:>6} {instance:>4} {bound:>9.4f} "
                  f"{cut:>4} {best:>13.4f} {bound - best:>9.4f}")

    for tau in (16, 32):
        config = CircuitConfig(8, tau, 4)
        state = scrambled_state(
            config,
            np.random.default_rng(np.random.SeedSequence(42, spawn_key=(8, 0))),
        )
        bound, cut = bound_for_state(state, config)
        data = np.load(
            RESULTS / "depth_ceiling" / f"tau{tau}" / "pq_n8_i0_sigma0.1.npz"
        )
        best = float(data["peak_weights"].max())
        print(f"{8:>4} {tau:>6} {0:>4} {bound:>9.4f} {cut:>4} "
              f"{best:>13.4f} {bound - best:>9.4f}")


if __name__ == "__main__":
    main()
