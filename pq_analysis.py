"""Analysis of the P(q) ensembles produced by pq_experiment.py.

Reads every ``pq_*.npz`` in the results directory and produces:

  * raw P(q) histograms per (n, instance), with the decorrelated-residual
    floor q ~ delta_i * delta_j marked -- two solutions that share nothing
    but the target string already overlap at that level;
  * normalized P(q_hat) histograms, q_hat = (q - d_i d_j) / (1 - d_i d_j),
    the across-n comparable version (the floor moves by an order of
    magnitude over the grid);
  * the intermediate-band mass q_hat in [0.25, 0.75] versus n, the simplest
    gap statistic: clustering with a gap empties this band;
  * the fraction of restarts below the near-optimality filter versus n,
    which is the trainability curve;
  * the init-scale control comparison (sigma = 0.1 protocol vs dispersed),
    whenever both runs exist for the same (n, instance);
  * the reachable peakedness versus the paper's 1.189^-n law
    (arXiv:2404.14493, Section 3).

All pair counts come from ~200 independent solutions, so histograms are read
qualitatively; no fine-grained error bars are drawn on purpose.

Usage:
    python pq_analysis.py                          # reads results/pq
    python pq_analysis.py --results results/pq --epsilon 0.2
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

PROTOCOL_INIT_SCALE = 0.1
EPSILON_GRID = (0.1, 0.2, 0.3)
BAND = (0.25, 0.75)
INSTANCE_COLORS = ("#1f77b4", "#d62728", "#2ca02c")


@dataclass(frozen=True)
class Ensemble:
    """One saved (n, instance, init-scale) solution ensemble."""

    num_qubits: int
    instance_index: int
    init_scale: float
    peak_weights: np.ndarray
    overlaps: np.ndarray
    num_steps: np.ndarray
    baseline_peak_weight: float

    @property
    def label(self) -> str:
        return (
            f"n={self.num_qubits} i={self.instance_index} "
            f"sigma={self.init_scale:g}"
        )


def load_ensembles(directory: Path) -> list[Ensemble]:
    ensembles = []
    for path in sorted(directory.glob("pq_*.npz")):
        with np.load(path) as data:
            ensembles.append(
                Ensemble(
                    num_qubits=int(data["num_qubits"]),
                    instance_index=int(data["instance_index"]),
                    init_scale=float(data["init_scale"]),
                    peak_weights=data["peak_weights"],
                    overlaps=data["overlap_matrix"],
                    num_steps=data["num_steps"],
                    baseline_peak_weight=float(data["baseline_peak_weight"]),
                )
            )
        logger.info("loaded %s", path.name)
    return ensembles


def near_optimal_mask(peak_weights: np.ndarray, epsilon: float) -> np.ndarray:
    """Restarts within a factor (1 - epsilon) of the instance's best delta."""
    return peak_weights >= (1.0 - epsilon) * peak_weights.max()


def pair_overlaps(
    ensemble: Ensemble, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangle (q, q_hat, floor) over near-optimal solution pairs."""
    kept = np.flatnonzero(near_optimal_mask(ensemble.peak_weights, epsilon))
    overlaps = ensemble.overlaps[np.ix_(kept, kept)]
    deltas = ensemble.peak_weights[kept]
    floor = np.outer(deltas, deltas)
    normalized = (overlaps - floor) / (1.0 - floor)
    rows, cols = np.triu_indices(len(kept), k=1)
    return overlaps[rows, cols], normalized[rows, cols], floor[rows, cols]


def band_mass(normalized: np.ndarray) -> float:
    """Fraction of pairs whose q_hat falls in the intermediate band."""
    if normalized.size == 0:
        return float("nan")
    inside = (normalized >= BAND[0]) & (normalized <= BAND[1])
    return float(np.mean(inside))


def group_protocol(ensembles: Sequence[Ensemble]) -> dict[int, list[Ensemble]]:
    """Protocol-scale ensembles grouped by n, instances sorted."""
    grouped: dict[int, list[Ensemble]] = {}
    for ensemble in ensembles:
        if np.isclose(ensemble.init_scale, PROTOCOL_INIT_SCALE):
            grouped.setdefault(ensemble.num_qubits, []).append(ensemble)
    for members in grouped.values():
        members.sort(key=lambda e: e.instance_index)
    return dict(sorted(grouped.items()))


def plot_raw(grouped: dict[int, list[Ensemble]], epsilon: float, out: Path) -> None:
    rows = len(grouped)
    cols = max(len(members) for members in grouped.values())
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False, sharex=True
    )
    bins = np.linspace(0.0, 1.0, 51)
    for row, (num_qubits, members) in enumerate(grouped.items()):
        for col in range(cols):
            axis = axes[row][col]
            if col >= len(members):
                axis.axis("off")
                continue
            ensemble = members[col]
            raw, _, floor = pair_overlaps(ensemble, epsilon)
            axis.hist(raw, bins=bins, color="#1f77b4", alpha=0.85)
            axis.axvline(
                float(np.mean(floor)), color="#d62728", linestyle="--", linewidth=1
            )
            axis.set_title(
                f"n={num_qubits}, instance {ensemble.instance_index}", fontsize=9
            )
            axis.set_yticks([])
        axes[row][0].set_ylabel("pairs")
    axes[-1][0].set_xlabel("q")
    fig.suptitle(
        "Raw pairwise state overlaps between near-optimal solutions "
        f"(filter: delta >= {1 - epsilon:g} x best; dashed: mean floor "
        "delta_i delta_j)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "pq_raw.png", dpi=200)
    plt.close(fig)


def plot_normalized(
    grouped: dict[int, list[Ensemble]], epsilon: float, out: Path
) -> None:
    fig, axes = plt.subplots(
        1, len(grouped), figsize=(3.4 * len(grouped), 3.0), squeeze=False, sharey=True
    )
    bins = np.linspace(-0.1, 1.0, 56)
    for col, (num_qubits, members) in enumerate(grouped.items()):
        axis = axes[0][col]
        for ensemble in members:
            _, normalized, _ = pair_overlaps(ensemble, epsilon)
            axis.hist(
                normalized,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.4,
                color=INSTANCE_COLORS[ensemble.instance_index % len(INSTANCE_COLORS)],
                label=f"instance {ensemble.instance_index}",
            )
        axis.axvspan(*BAND, color="0.85", zorder=0)
        axis.set_title(f"n = {num_qubits}", fontsize=10)
        axis.set_xlabel(r"$\hat{q}$")
        if col == 0:
            axis.set_ylabel(r"$P(\hat{q})$")
            axis.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Floor-normalized overlap distribution by system size "
        "(shaded: intermediate band; ~200 solutions per panel, read "
        "qualitatively)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "pq_normalized.png", dpi=200)
    plt.close(fig)


def plot_statistics(grouped: dict[int, list[Ensemble]], out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    sizes = list(grouped)
    for epsilon, style in zip(EPSILON_GRID, ("-", "--", ":"), strict=True):
        masses = [
            np.nanmean(
                [band_mass(pair_overlaps(e, epsilon)[1]) for e in grouped[n]]
            )
            for n in sizes
        ]
        stalled = [
            np.mean(
                [
                    1.0 - np.mean(near_optimal_mask(e.peak_weights, epsilon))
                    for e in grouped[n]
                ]
            )
            for n in sizes
        ]
        left.plot(sizes, masses, style, marker="o", label=f"eps = {epsilon:g}")
        right.plot(sizes, stalled, style, marker="o", label=f"eps = {epsilon:g}")
    left.set_xlabel("n")
    left.set_ylabel(f"pair mass in band {BAND}")
    left.set_title("Intermediate-band mass (gap statistic)", fontsize=10)
    left.legend(fontsize=8, frameon=False)
    right.set_xlabel("n")
    right.set_ylabel("fraction of restarts filtered out")
    right.set_title("Restarts below the near-optimality filter", fontsize=10)
    right.legend(fontsize=8, frameon=False)
    for axis in (left, right):
        axis.set_xticks(sizes)
    fig.tight_layout()
    fig.savefig(out / "pq_statistics.png", dpi=200)
    plt.close(fig)


def plot_init_control(
    ensembles: Sequence[Ensemble], epsilon: float, out: Path
) -> bool:
    """Overlay protocol vs dispersed-init P(q_hat); returns True if drawn."""
    by_key: dict[tuple[int, int], list[Ensemble]] = {}
    for ensemble in ensembles:
        by_key.setdefault(
            (ensemble.num_qubits, ensemble.instance_index), []
        ).append(ensemble)
    pairs = {key: members for key, members in by_key.items() if len(members) > 1}
    if not pairs:
        return False
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(4.0 * len(pairs), 3.2), squeeze=False
    )
    bins = np.linspace(-0.1, 1.0, 56)
    for col, ((num_qubits, instance), members) in enumerate(sorted(pairs.items())):
        axis = axes[0][col]
        for ensemble in sorted(members, key=lambda e: e.init_scale):
            _, normalized, _ = pair_overlaps(ensemble, epsilon)
            axis.hist(
                normalized,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.4,
                label=f"init sigma = {ensemble.init_scale:g}",
            )
        axis.set_title(f"n = {num_qubits}, instance {instance}", fontsize=10)
        axis.set_xlabel(r"$\hat{q}$")
        axis.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel(r"$P(\hat{q})$")
    fig.suptitle("Initialization control (same circuit)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "pq_init_control.png", dpi=200)
    plt.close(fig)
    return True


def plot_delta_scaling(grouped: dict[int, list[Ensemble]], out: Path) -> None:
    """Reachable peakedness versus n, fitted on the paper's own statistic.

    The fitted series is the per-instance best of the restart batch,
    averaged over instances -- exactly what arXiv:2404.14493 reports
    (Section 4: batches of random restarts, best of the batch, averaged
    over instances). The unfiltered ensemble mean is shown for reference;
    fitting it instead yields a steeper base (1.220 on this grid) because
    the growing stalled fraction (fact F3) drags the mean down at large n.
    The base uncertainty is a bootstrap over the instances of each size.
    """
    sizes = np.array(list(grouped))
    means = np.array(
        [np.mean([e.peak_weights.mean() for e in grouped[n]]) for n in sizes]
    )
    best_lists = [
        np.array([e.peak_weights.max() for e in grouped[n]]) for n in sizes
    ]
    bests = np.array([row.mean() for row in best_lists])
    fig, axis = plt.subplots(figsize=(4.6, 3.4))
    axis.semilogy(sizes, means, "o-", label="ensemble mean delta")
    axis.semilogy(sizes, bests, "s--", label="best delta (mean over instances)")
    if len(sizes) > 1:
        slope, intercept = np.polyfit(sizes, np.log(bests), 1)
        base = float(np.exp(-slope))
        bootstrap = np.random.default_rng(2026)
        resampled = np.empty(1000)
        for index in range(len(resampled)):
            draws = [
                np.mean(bootstrap.choice(row, size=len(row)))
                for row in best_lists
            ]
            resampled[index] = np.exp(-np.polyfit(sizes, np.log(draws), 1)[0])
        spread = float(np.std(resampled))
        mean_base = float(np.exp(-np.polyfit(sizes, np.log(means), 1)[0]))
        print(
            f"reachable-peakedness fit (per-instance best, averaged over "
            f"instances): base = {base:.3f} +/- {spread:.3f} (bootstrap); "
            f"ensemble-mean fit for reference: {mean_base:.3f}"
        )
        axis.semilogy(
            sizes,
            np.exp(intercept + slope * sizes),
            ":",
            color="0.4",
            label=f"fit {base:.3f}$^{{-n}}$ (paper: 1.189$^{{-n}}$)",
        )
    axis.set_xlabel("n")
    axis.set_ylabel("peakedness delta")
    axis.set_xticks(sizes)
    axis.set_title("Reachable peakedness vs the paper's law", fontsize=10)
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "delta_scaling.png", dpi=200)
    plt.close(fig)


def print_summary(ensembles: Sequence[Ensemble], epsilon: float) -> None:
    header = (
        f"{'ensemble':>24}  {'best':>7}  {'mean':>7}  {'stalled':>7}  "
        f"{'band':>6}  {'med qhat':>8}"
    )
    print(header)
    print("-" * len(header))
    for ensemble in ensembles:
        _, normalized, _ = pair_overlaps(ensemble, epsilon)
        stalled = 1.0 - float(np.mean(near_optimal_mask(ensemble.peak_weights, epsilon)))
        median = float(np.median(normalized)) if normalized.size else float("nan")
        print(
            f"{ensemble.label:>24}  {ensemble.peak_weights.max():7.4f}  "
            f"{ensemble.peak_weights.mean():7.4f}  {stalled:7.2%}  "
            f"{band_mass(normalized):6.3f}  {median:8.3f}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", type=str, default="results/pq", help="directory of pq_*.npz"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="near-optimality filter: keep delta >= (1 - epsilon) x best",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    args = parse_args(argv)
    directory = Path(args.results)
    ensembles = load_ensembles(directory)
    if not ensembles:
        raise SystemExit(f"no pq_*.npz files found in {directory}")

    grouped = group_protocol(ensembles)
    if grouped:
        plot_raw(grouped, args.epsilon, directory)
        plot_normalized(grouped, args.epsilon, directory)
        plot_statistics(grouped, directory)
        plot_delta_scaling(grouped, directory)
        logger.info("wrote pq_raw / pq_normalized / pq_statistics / delta_scaling")
    if plot_init_control(ensembles, args.epsilon, directory):
        logger.info("wrote pq_init_control")
    print_summary(ensembles, args.epsilon)


if __name__ == "__main__":
    main()
