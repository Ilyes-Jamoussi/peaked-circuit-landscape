"""P(q) measurement on the peaked-circuit optimization landscape.

First experiment of the research program in README.md (Section 3.4): for a
fixed random-circuit instance, run many independently initialized
optimizations of the peaking layers and record *every* final solution, not
just the best of the batch. The pairwise overlap distribution P(q) of the
resulting solution ensemble is the standard empirical diagnostic for
solution-space clustering (Gamarnik, arXiv:2109.14409), measured here, to our
knowledge for the first time, on the circuit-synthesis landscape of Aaronson
& Zhang (arXiv:2404.14493).

Protocol, per random-circuit instance:

  * geometry follows the paper's scaling regime tau_r = n, tau_p = n/2, the
    regime of the 1.189^-n law (arXiv:2404.14493, Section 3, Fig. 3c);
  * ``--restarts`` independent Adam runs with the reproduction's
    hyperparameters, each from its own initialization
    theta ~ N(0, ``--init-scale``); the default scale 0.1 matches the
    reproduction protocol, and larger scales serve as a control against
    clustering artifacts of the common near-identity starting region;
  * q_ij = |<psi_i|psi_j>|^2 for every pair of final output states, stored
    as a dense overlap matrix (the states themselves can be rebuilt from the
    saved angles).

Builds on the reproduction, expected as a sibling directory or at
$PEAKED_REPRO_PATH: https://github.com/Ilyes-Jamoussi/peaked-circuits-pennylane

Usage:
    python pq_experiment.py --pilot                     # timing estimate
    python pq_experiment.py --num-qubits 8 --instance 0
    python pq_experiment.py --num-qubits 12 --instance 0 --init-scale 1.0
"""

from __future__ import annotations

import os

# One BLAS thread per process: restarts are parallelized at the process
# level, and oversubscribed threads slow every worker down.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import logging
import multiprocessing
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

def _locate_repro() -> Path:
    """Reproduction repo: PEAKED_REPRO_PATH, else the nearest enclosing sibling."""
    override = os.environ.get("PEAKED_REPRO_PATH")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for base in here.parents:
        candidate = base / "peaked-circuits-pennylane"
        if candidate.is_dir():
            return candidate
    return here.parent.parent / "peaked-circuits-pennylane"


REPRO_PATH = _locate_repro()
sys.path.insert(0, str(REPRO_PATH))

import pennylane as qml  # noqa: E402
from pennylane import numpy as pnp  # noqa: E402

from peaked_circuits import (  # noqa: E402
    CircuitConfig,
    _apply_peaking_section,
    _apply_random_section,
    build_peak_weight_fn,
    random_section_peak_weight,
    sample_haar_random_layers,
)

logger = logging.getLogger(__name__)

#: n values of the first measurement pass; within the range of the paper's
#: Fig. 3c (n = 8 to 16), so the delta scaling can be cross-checked.
GRID_QUBIT_COUNTS = (8, 10, 12, 14)


def az_scaling_config(
    num_qubits: int,
    num_random_layers: int | None = None,
    num_peaking_layers: int | None = None,
) -> CircuitConfig:
    """Circuit geometry of the 1.189^-n regime: tau_r = n, tau_p = n/2."""
    if num_random_layers is None:
        num_random_layers = num_qubits
    if num_peaking_layers is None:
        if num_qubits % 2:
            raise ValueError("tau_p = n/2 requires an even number of qubits.")
        num_peaking_layers = num_qubits // 2
    return CircuitConfig(num_qubits, num_random_layers, num_peaking_layers)


@dataclass(frozen=True)
class RestartSettings:
    """Per-restart Adam settings; defaults match the reproduction protocol.

    ``max_steps`` defaults to 400 rather than the reproduction's 2000: P(q)
    needs an ensemble of near-optimal solutions, not the polished optimum,
    and the reproduction's log shows the peak weight plateauing there.
    """

    max_steps: int = 400
    learning_rate: float = 0.05
    decay_every: int = 300
    decay_factor: float = 0.5
    convergence_tol: float = 1e-8
    min_steps: int = 100
    init_scale: float = 0.1
    optimizer: str = "adam"  # robustness matrix (2026-07-17); default = protocol

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.learning_rate <= 0:
            raise ValueError("max_steps and learning_rate must be positive.")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0.")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'.")


@dataclass(frozen=True)
class RestartOutcome:
    """Final solution of one independently initialized Adam run."""

    theta_init: np.ndarray
    theta_final: np.ndarray
    peak_weight: float
    num_steps: int
    seconds: float


@dataclass(frozen=True)
class RestartTask:
    """Everything a worker process needs to run one restart."""

    circuit_config: CircuitConfig
    haar_layers: list
    settings: RestartSettings
    base_seed: int
    instance_index: int
    restart_index: int


def restart_rng(task: RestartTask) -> np.random.Generator:
    """Deterministic per-restart generator, independent of scheduling.

    The spawn key includes the init scale so that runs at different scales
    draw statistically independent initializations rather than radially
    rescaled copies of one another.
    """
    return np.random.default_rng(
        np.random.SeedSequence(
            task.base_seed,
            spawn_key=(
                task.circuit_config.num_qubits,
                task.instance_index,
                task.restart_index,
                int(round(task.settings.init_scale * 1000)),
            ),
        )
    )


def optimize_one_restart(
    peak_weight_fn,
    num_parameters: int,
    settings: RestartSettings,
    rng: np.random.Generator,
) -> RestartOutcome:
    """One Adam run from a fresh initialization; returns its final solution.

    Same loop as the reproduction's ``maximize_peak_weight`` (learning-rate
    step decay, early stopping on the per-step cost change), except that the
    solution is returned unconditionally instead of competing in a batch.
    """
    theta_init = rng.normal(0.0, settings.init_scale, num_parameters)
    theta = pnp.array(theta_init, requires_grad=True)
    if settings.optimizer == "adam":
        optimizer = qml.AdamOptimizer(stepsize=settings.learning_rate)
    else:
        optimizer = qml.GradientDescentOptimizer(stepsize=settings.learning_rate)

    def cost(parameters):
        return -peak_weight_fn(parameters)

    start = time.perf_counter()
    previous_cost = np.inf
    steps_taken = settings.max_steps
    for step in range(1, settings.max_steps + 1):
        theta, current_cost = optimizer.step_and_cost(cost, theta)
        current_cost = float(current_cost)
        if (
            step > settings.min_steps
            and abs(previous_cost - current_cost) < settings.convergence_tol
        ):
            steps_taken = step
            break
        previous_cost = current_cost
        if step % settings.decay_every == 0:
            optimizer.stepsize *= settings.decay_factor

    return RestartOutcome(
        theta_init=theta_init,
        theta_final=np.asarray(theta),
        peak_weight=float(peak_weight_fn(theta)),
        num_steps=steps_taken,
        seconds=time.perf_counter() - start,
    )


def run_restart_task(task: RestartTask) -> tuple[int, RestartOutcome]:
    """Worker entry point: rebuild the circuit and run one restart."""
    peak_weight_fn = build_peak_weight_fn(task.circuit_config, task.haar_layers)
    outcome = optimize_one_restart(
        peak_weight_fn,
        task.circuit_config.num_peaking_parameters,
        task.settings,
        restart_rng(task),
    )
    return task.restart_index, outcome


def build_state_fn(config: CircuitConfig, haar_layers: list):
    """Return theta -> C(theta)|0^n>, the full output state, for overlaps."""
    device = qml.device("default.qubit", wires=config.num_qubits)

    @qml.qnode(device)
    def state_fn(theta):
        _apply_random_section(config, haar_layers)
        _apply_peaking_section(config, theta)
        return qml.state()

    return state_fn


def overlap_matrix(states: np.ndarray) -> np.ndarray:
    """Pairwise overlaps q_ij = |<psi_i|psi_j>|^2 of stacked states (R, 2^n)."""
    gram = states @ states.conj().T
    return np.abs(gram) ** 2


@dataclass(frozen=True)
class InstanceData:
    """Solution ensemble of one circuit instance, ready to save."""

    thetas_init: np.ndarray
    thetas_final: np.ndarray
    peak_weights: np.ndarray
    num_steps: np.ndarray
    restart_seconds: np.ndarray
    argmax_indices: np.ndarray
    argmax_weights: np.ndarray
    overlaps: np.ndarray
    baseline_peak_weight: float
    elapsed_seconds: float


def run_instance(
    circuit_config: CircuitConfig,
    settings: RestartSettings,
    num_restarts: int,
    base_seed: int,
    instance_index: int,
    num_workers: int,
) -> InstanceData:
    """Fix one random circuit and collect ``num_restarts`` final solutions."""
    circuit_rng = np.random.default_rng(
        np.random.SeedSequence(
            base_seed, spawn_key=(circuit_config.num_qubits, instance_index)
        )
    )
    haar_layers = sample_haar_random_layers(circuit_config, circuit_rng)
    baseline = random_section_peak_weight(circuit_config, haar_layers)
    logger.info(
        "n = %d, tau_r = %d, tau_p = %d, instance %d: %d restarts, "
        "init scale %g, baseline peak weight %.2e",
        circuit_config.num_qubits,
        circuit_config.num_random_layers,
        circuit_config.num_peaking_layers,
        instance_index,
        num_restarts,
        settings.init_scale,
        baseline,
    )

    tasks = [
        RestartTask(
            circuit_config=circuit_config,
            haar_layers=haar_layers,
            settings=settings,
            base_seed=base_seed,
            instance_index=instance_index,
            restart_index=index,
        )
        for index in range(num_restarts)
    ]

    outcomes: list[RestartOutcome | None] = [None] * num_restarts
    start = time.perf_counter()
    if num_workers == 1:
        for done, task in enumerate(tasks, start=1):
            index, outcome = run_restart_task(task)
            outcomes[index] = outcome
            _log_restart(done, num_restarts, index, outcome)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as pool:
            futures = [pool.submit(run_restart_task, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                index, outcome = future.result()
                outcomes[index] = outcome
                _log_restart(done, num_restarts, index, outcome)
    elapsed = time.perf_counter() - start

    complete = [outcome for outcome in outcomes if outcome is not None]
    assert len(complete) == num_restarts
    peak_weights = np.array([outcome.peak_weight for outcome in complete])

    state_fn = build_state_fn(circuit_config, haar_layers)
    states = np.stack(
        [np.asarray(state_fn(outcome.theta_final)) for outcome in complete]
    )
    zero_string_weights = np.abs(states[:, 0]) ** 2
    if not np.allclose(zero_string_weights, peak_weights, atol=1e-9):
        logger.warning(
            "state amplitudes disagree with recorded peak weights (max diff %.2e)",
            float(np.max(np.abs(zero_string_weights - peak_weights))),
        )
    probabilities = np.abs(states) ** 2
    argmax_indices = probabilities.argmax(axis=1)
    argmax_weights = probabilities[np.arange(len(complete)), argmax_indices]

    logger.info(
        "instance done in %.1f s: delta best %.4f, mean %.4f, min %.4f; "
        "%d/%d restarts peak on the 0^n string",
        elapsed,
        peak_weights.max(),
        peak_weights.mean(),
        peak_weights.min(),
        int(np.sum(argmax_indices == 0)),
        num_restarts,
    )
    return InstanceData(
        thetas_init=np.stack([outcome.theta_init for outcome in complete]),
        thetas_final=np.stack([outcome.theta_final for outcome in complete]),
        peak_weights=peak_weights,
        num_steps=np.array([outcome.num_steps for outcome in complete]),
        restart_seconds=np.array([outcome.seconds for outcome in complete]),
        argmax_indices=argmax_indices,
        argmax_weights=argmax_weights,
        overlaps=overlap_matrix(states),
        baseline_peak_weight=baseline,
        elapsed_seconds=elapsed,
    )


def _log_restart(done: int, total: int, index: int, outcome: RestartOutcome) -> None:
    logger.info(
        "restart %3d finished (%3d/%d)  delta = %.4f  steps = %d  [%.1f s]",
        index,
        done,
        total,
        outcome.peak_weight,
        outcome.num_steps,
        outcome.seconds,
    )


def save_instance(
    path: Path,
    circuit_config: CircuitConfig,
    settings: RestartSettings,
    data: InstanceData,
    base_seed: int,
    instance_index: int,
) -> None:
    """Persist the ensemble and everything needed to rebuild it exactly."""
    np.savez_compressed(
        path,
        thetas_init=data.thetas_init,
        thetas_final=data.thetas_final,
        peak_weights=data.peak_weights,
        num_steps=data.num_steps,
        restart_seconds=data.restart_seconds,
        argmax_indices=data.argmax_indices,
        argmax_weights=data.argmax_weights,
        overlap_matrix=data.overlaps,
        baseline_peak_weight=data.baseline_peak_weight,
        elapsed_seconds=data.elapsed_seconds,
        num_qubits=circuit_config.num_qubits,
        num_random_layers=circuit_config.num_random_layers,
        num_peaking_layers=circuit_config.num_peaking_layers,
        base_seed=base_seed,
        instance_index=instance_index,
        max_steps=settings.max_steps,
        learning_rate=settings.learning_rate,
        decay_every=settings.decay_every,
        decay_factor=settings.decay_factor,
        convergence_tol=settings.convergence_tol,
        min_steps=settings.min_steps,
        init_scale=settings.init_scale,
    )
    logger.info("saved instance data to %s", path)


def run_pilot(base_seed: int, num_workers: int, num_restarts: int = 2) -> None:
    """Measure seconds/step at each grid size and extrapolate the budget.

    Runs serially for clean timing; parallel workers share memory bandwidth,
    so the extrapolation is a lower bound on the real wall time.
    """
    pilot_steps = 60
    production = RestartSettings()
    total_hours = 0.0
    for num_qubits in GRID_QUBIT_COUNTS:
        config = az_scaling_config(num_qubits)
        settings = RestartSettings(max_steps=pilot_steps, min_steps=pilot_steps)
        circuit_rng = np.random.default_rng(
            np.random.SeedSequence(base_seed, spawn_key=(num_qubits, 0))
        )
        haar_layers = sample_haar_random_layers(config, circuit_rng)
        peak_weight_fn = build_peak_weight_fn(config, haar_layers)
        seconds_per_step = []
        for index in range(num_restarts):
            task = RestartTask(config, haar_layers, settings, base_seed, 0, index)
            outcome = optimize_one_restart(
                peak_weight_fn, config.num_peaking_parameters, settings, restart_rng(task)
            )
            seconds_per_step.append(outcome.seconds / outcome.num_steps)
        step_time = float(np.mean(seconds_per_step))
        restart_minutes = step_time * production.max_steps / 60
        instance_hours = restart_minutes * 200 / num_workers / 60
        total_hours += 3 * instance_hours
        logger.info(
            "n = %2d: %.3f s/step -> ~%.1f min/restart (%d steps), "
            "~%.2f h/instance (200 restarts, %d workers)",
            num_qubits,
            step_time,
            restart_minutes,
            production.max_steps,
            instance_hours,
            num_workers,
        )
    logger.info(
        "full grid estimate (3 instances per n): ~%.1f h wall time, "
        "lower bound (serial timing, shared memory bandwidth)",
        total_hours,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="measure seconds/step at each grid size and estimate the budget",
    )
    parser.add_argument("--num-qubits", type=int, default=None, help="n (even)")
    parser.add_argument(
        "--instance", type=int, default=0, help="circuit instance index"
    )
    parser.add_argument(
        "--restarts", type=int, default=200, help="independent Adam runs"
    )
    parser.add_argument(
        "--max-steps", type=int, default=400, help="Adam steps per restart"
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.1,
        help="std-dev of the initialization (0.1 = protocol, larger = control)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="parallel worker processes"
    )
    parser.add_argument("--base-seed", type=int, default=42, help="root RNG seed")
    parser.add_argument(
        "--random-layers",
        type=int,
        default=None,
        help="override tau_r (default: n, the A&Z scaling regime)",
    )
    parser.add_argument(
        "--peaking-layers",
        type=int,
        default=None,
        help="override tau_p (default: n/2, the A&Z scaling regime)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.05,
        help="Adam/SGD stepsize (0.05 = protocol; robustness matrix)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=("adam", "sgd"),
        help="optimizer (adam = protocol; robustness matrix)",
    )
    parser.add_argument(
        "--output", type=str, default="results/pq", help="output directory"
    )
    args = parser.parse_args(argv)
    if not args.pilot and args.num_qubits is None:
        parser.error("--num-qubits is required unless --pilot is given")
    if args.restarts < 2:
        parser.error("--restarts must be >= 2 (P(q) needs pairs)")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    args = parse_args(argv)
    if args.pilot:
        run_pilot(args.base_seed, args.workers)
        return

    circuit_config = az_scaling_config(
        args.num_qubits, args.random_layers, args.peaking_layers
    )
    settings = RestartSettings(
        max_steps=args.max_steps,
        init_scale=args.init_scale,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
    )
    data = run_instance(
        circuit_config,
        settings,
        num_restarts=args.restarts,
        base_seed=args.base_seed,
        instance_index=args.instance,
        num_workers=args.workers,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"pq_n{circuit_config.num_qubits}_i{args.instance}"
        f"_sigma{args.init_scale:g}.npz"
    )
    save_instance(path, circuit_config, settings, data, args.base_seed, args.instance)


if __name__ == "__main__":
    main()
