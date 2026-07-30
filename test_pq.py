"""Sanity tests for pq_experiment.py.

Run with
    python test_pq.py
or, if pytest is installed,
    pytest test_pq.py
"""

from __future__ import annotations

import numpy as np

from pq_experiment import (
    RestartSettings,
    az_scaling_config,
    build_state_fn,
    overlap_matrix,
    run_instance,
)
from peaked_circuits import (
    CircuitConfig,
    build_peak_weight_fn,
    sample_haar_random_layers,
)

TINY_SETTINGS = RestartSettings(max_steps=30, min_steps=5)


def tiny_instance(num_workers: int = 1, base_seed: int = 7):
    """A cheap ensemble: n = 4, tau_r = 4, tau_p = 2, three restarts."""
    return run_instance(
        az_scaling_config(4),
        TINY_SETTINGS,
        num_restarts=3,
        base_seed=base_seed,
        instance_index=0,
        num_workers=num_workers,
    )


def test_az_scaling_config() -> None:
    config = az_scaling_config(12)
    assert (config.num_qubits, config.num_random_layers, config.num_peaking_layers) == (
        12,
        12,
        6,
    )
    try:
        az_scaling_config(9)
    except ValueError:
        pass
    else:
        raise AssertionError("odd n must be rejected when tau_p defaults to n/2")
    override = az_scaling_config(9, num_peaking_layers=4)
    assert override.num_peaking_layers == 4  # explicit tau_p bypasses the check


def test_overlap_matrix_properties() -> None:
    data = tiny_instance()
    overlaps = data.overlaps
    assert overlaps.shape == (3, 3)
    assert np.allclose(overlaps, overlaps.T)
    assert np.allclose(np.diag(overlaps), 1.0)
    assert np.all(overlaps >= 0.0) and np.all(overlaps <= 1.0 + 1e-12)
    assert np.all(data.peak_weights > data.baseline_peak_weight)


def test_determinism() -> None:
    first, second = tiny_instance(), tiny_instance()
    assert np.array_equal(first.thetas_init, second.thetas_init)
    assert np.array_equal(first.thetas_final, second.thetas_final)
    assert np.array_equal(first.overlaps, second.overlaps)


def test_worker_invariance() -> None:
    """Parallel and serial execution must produce identical ensembles."""
    serial, parallel = tiny_instance(num_workers=1), tiny_instance(num_workers=2)
    assert np.array_equal(serial.peak_weights, parallel.peak_weights)
    assert np.array_equal(serial.thetas_final, parallel.thetas_final)
    assert np.array_equal(serial.overlaps, parallel.overlaps)


def test_state_matches_projector() -> None:
    """The state-vector qnode and the projector qnode must agree on delta."""
    config = CircuitConfig(num_qubits=4, num_random_layers=3, num_peaking_layers=2)
    haar_layers = sample_haar_random_layers(config, np.random.default_rng(3))
    theta = np.random.default_rng(4).normal(0.0, 0.3, config.num_peaking_parameters)
    delta = float(build_peak_weight_fn(config, haar_layers)(theta))
    state = np.asarray(build_state_fn(config, haar_layers)(theta))
    assert np.isclose(delta, np.abs(state[0]) ** 2)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_overlap_matrix_of_known_states() -> None:
    """Overlaps of hand-built states come out exactly right."""
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    states = np.stack([np.array([1.0, 0.0]), np.array([0.0, 1.0]), plus]).astype(
        complex
    )
    expected = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]])
    assert np.allclose(overlap_matrix(states), expected)


def test_restart_settings_validation() -> None:
    for invalid in (
        lambda: RestartSettings(max_steps=0),
        lambda: RestartSettings(learning_rate=0.0),
        lambda: RestartSettings(init_scale=-1.0),
    ):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, fn in tests:
        fn()
        print(f"{name}: OK")
    print(f"all {len(tests)} tests passed")
