"""Consolidated figures for the manuscript (figures/*.pdf).

Panels are generated from the restart ensembles under results/ (18
instances per size at n = 8-14, 4 at n = 16), the budget-800
ensembles and the resolution-matched connectivity archives, together
with the moment-scan logs committed under analysis/ (values cited
inline with their log of origin). The ensembles are not part of this
repository; they are archived separately, see README.

Palette: Okabe-Ito subset in fixed n-order, validated for CVD
separation; every series carries a distinct marker as secondary
encoding, and identity is direct-labeled where space allows.

Usage:  python figures/make_figures.py     (from the repository root)
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

# Fixed timestamp so regenerated PDFs are byte-identical to the committed
# ones: matplotlib otherwise stamps the current date into every file.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")  # 2026-01-01 UTC

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from budget_scan import expected_max  # noqa: E402
from pq_analysis import Ensemble, near_optimal_mask, pair_overlaps  # noqa: E402

OUT = Path(__file__).resolve().parent

SIZES = (8, 10, 12, 14, 16)
COLOR = {8: "#0072B2", 10: "#E69F00", 12: "#009E73",
         14: "#D55E00", 16: "#CC79A7"}
MARKER = {8: "o", 10: "s", 12: "D", 14: "^", 16: "v"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "figure.dpi": 150,
})


def load_bests() -> dict[int, np.ndarray]:
    bests: dict[int, list[float]] = {}
    for name in sorted(glob.glob(str(ROOT / "results/pq/pq_n*_sigma0.1.npz"))):
        match = re.search(r"pq_n(\d+)_i(\d+)_", name)
        values = np.load(name)["peak_weights"]
        if len(values) < 200:
            continue
        bests.setdefault(int(match.group(1)), []).append(float(values.max()))
    return {n: np.array(v) for n, v in bests.items()}


def fig_scaling() -> None:
    """Mean-of-best vs n with the steepening law (the headline figure)."""
    bests = load_bests()
    sizes = np.array(sorted(bests))
    mean = np.array([bests[n].mean() for n in sizes])
    sem = np.array([bests[n].std(ddof=1) / np.sqrt(len(bests[n]))
                    for n in sizes])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [3, 2]}
    )

    # Left: the decay with the n <= 14 fixed-base extrapolation.
    weights = mean / sem  # 1/sigma of ln(mean)
    head = np.polyfit(sizes[:-1], np.log(mean[:-1]), 1, w=weights[:-1])
    grid = np.linspace(7.5, 16.5, 50)
    ax.plot(grid, np.exp(np.polyval(head, grid)), "--", color="#888888",
            label=r"fixed-base fit ($n \leq 14$)")
    ax.errorbar(sizes, mean, yerr=sem, fmt="none", ecolor="#444444",
                elinewidth=1, capsize=2, zorder=3)
    for n in sizes:
        ax.plot(n, bests[n].mean(), MARKER[n], color=COLOR[n],
                markersize=6, zorder=4)
        ax.plot([n] * len(bests[n]), bests[n], ".", color=COLOR[n],
                alpha=0.25, markersize=3, zorder=2)
    extrapolated = float(np.exp(np.polyval(head, 16)))
    ax.annotate("", xy=(16, mean[-1] * 1.06), xytext=(16, extrapolated * 0.96),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1))
    ax.text(15.75, np.sqrt(mean[-1] * extrapolated),
            r"$5\sigma$", ha="right", va="center")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlabel(r"qubits $n$")
    ax.set_ylabel(r"mean-of-best $\overline{\delta}_{\mathrm{best}}$")
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    ax.grid(True, axis="y")

    # Right: local log-steps per two qubits (weight-free observable).
    steps = -np.diff(np.log(mean))
    mids = (sizes[:-1] + sizes[1:]) / 2
    step_err = np.sqrt((sem[:-1] / mean[:-1]) ** 2
                       + (sem[1:] / mean[1:]) ** 2)
    ax2.errorbar(mids, steps, yerr=step_err, fmt="o-", color="#0072B2",
                 markersize=5, capsize=2)
    ax2.axhline(2 * np.log(1.195), ls="--", color="#888888", lw=1)
    # Label the dashed reference line from below on the right, where the
    # steepening data curve leaves the region empty.
    ax2.text(15.2, 2 * np.log(1.195) - 0.012, r"$1.195^{-n}$ law",
             color="#666666", ha="right", va="top")
    ax2.set_xlabel(r"midpoint $n$")
    ax2.set_ylabel(r"$-\Delta \ln \overline{\delta}_{\mathrm{best}}$ per 2 qubits")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling.pdf")
    plt.close(fig)


def fig_pq() -> None:
    """Consolidated floor-normalized overlap distributions."""
    fig, axes = plt.subplots(1, 5, figsize=(6.9, 1.9), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, n in zip(axes, SIZES):
        pooled = []
        for name in sorted(
            glob.glob(str(ROOT / f"results/pq/pq_n{n}_i*_sigma0.1.npz"))
        ):
            data = np.load(name)
            ensemble = Ensemble(
                int(data["num_qubits"]), int(data["instance_index"]),
                float(data["init_scale"]), data["peak_weights"],
                data["overlap_matrix"], data["num_steps"],
                float(data["baseline_peak_weight"]),
            )
            _, normalized, _ = pair_overlaps(ensemble, 0.2)
            pooled.append(normalized)
        normalized = np.concatenate(pooled)
        ax.hist(normalized, bins=bins, color=COLOR[n], log=True,
                weights=np.full(len(normalized), 1.0 / len(normalized)))
        ax.set_title(rf"$n = {n}$", color=COLOR[n])
        ax.set_xlim(0, 1)
        ax.set_xticks((0, 0.5, 1))
        ax.set_xlabel(r"$\hat q$")
        if n == 8:
            ax.set_ylabel(r"pair fraction")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pq.pdf")
    plt.close(fig)


def fig_budget() -> None:
    """Exact expected best-of-B curves to B = 800."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharex=True)
    budgets = np.array([6, 12, 25, 50, 100, 200, 400, 800])
    for ax, n in zip(axes, (10, 12)):
        for i, ls in zip((0, 1, 2), ("-", "--", ":")):
            values = np.load(
                ROOT / f"results/budget800/pq_n{n}_i{i}_sigma0.1.npz"
            )["peak_weights"]
            curve = [expected_max(values, b) for b in budgets]
            ax.plot(budgets, curve, ls, color=COLOR[n], alpha=0.85,
                    label=rf"instance {i}")
        ax.set_xscale("log")
        ax.set_title(rf"$n = {n}$", color=COLOR[n])
        ax.set_xlabel(r"restart budget $B$")
        ax.grid(True, axis="y")
        ax.legend(frameon=False, fontsize=8, handlelength=1.8)
    axes[0].set_ylabel(r"$\mathbb{E}[\max_{B}\,\delta]$")
    fig.tight_layout()
    fig.savefig(OUT / "fig_budget.pdf")
    plt.close(fig)


def fig_corrugation() -> None:
    """Corrugation depth vs n at matched string resolution."""
    runs = [(8, "conn_n8_o150s1_m32.npz"), (10, "conn_n10_o150s1_m32.npz"),
            (12, "conn_n12_o150s1_m32.npz"), (16, "conn_n16_o150s1_m64.npz")]
    sizes, medians = [], []
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for n, name in runs:
        data = np.load(ROOT / f"results/connectivity/{name}")
        ratios = np.array([
            data[key][2] / min(data[key][0], data[key][1])
            for key in data.keys()
            if key.startswith("pair") and key.endswith("_scalars")
        ])
        sizes.append(n)
        medians.append(float(np.median(ratios)))
        ax.plot([n] * len(ratios), ratios, ".", color=COLOR[n],
                alpha=0.45, markersize=4)
        ax.plot(n, np.median(ratios), MARKER[n], color=COLOR[n],
                markersize=6, zorder=3)
    sizes = np.array(sizes)
    medians = np.array(medians)
    exp_fit = np.polyfit(sizes, np.log(medians), 1)
    pow_fit = np.polyfit(np.log(sizes), np.log(medians), 1)
    grid = np.linspace(7.5, 16.5, 60)
    ax.plot(grid, np.exp(np.polyval(exp_fit, grid)), "-", color="#444444",
            lw=1, label=rf"$\rho \propto {np.exp(exp_fit[0]):.3f}^{{\,n}}$")
    ax.plot(grid, np.exp(np.polyval(pow_fit, np.log(grid))), "--",
            color="#999999", lw=1,
            label=rf"$\rho \propto n^{{{pow_fit[0]:.1f}}}$")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlabel(r"qubits $n$")
    ax.set_ylabel(r"corrugation floor $\rho$")
    ax.legend(frameon=False, fontsize=8, loc="lower left", handlelength=1.6)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_corrugation.pdf")
    plt.close(fig)


def fig_moments() -> None:
    """Moment-hierarchy exponents vs depth (committed scan logs).

    Values from analysis/third_moment.log, analysis/fourth_moment.log
    (n = 6) and analysis/third_moment_n8.log, analysis/fourth_moment_n8.log
    (n = 8); the k = 3 exponents are ln m3 / ln m2 of those tables.
    """
    n6 = {  # tau_r: (m2, m3, m4)
        # tau_r = 2 is excluded: its certified truncation bound is 1.26 in
        # m4 units (16% of m4 = 7.901), above the 1% the caption asserts.
        # The n = 8 twin at tau_r = 2 is excluded for the same reason.
        1: (2.000, 5.207, 15.175),
        3: (1.391, 2.335, 4.375), 6: (1.118, 1.357, 1.752),
    }
    n8 = {
        1: (3.012, 14.717, 91.987), 3: (1.810, 4.737, 15.781),
        8: (1.164, 1.549, 2.320),
    }
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for data, n, color in ((n6, 6, "#56B4E9"), (n8, 8, COLOR[8])):
        depths = sorted(data)
        for k, ls, mk in ((3, "--", "s"), (4, "-", "o")):
            exponents = [np.log(data[t][k - 2]) / np.log(data[t][0])
                         for t in depths]
            ax.plot(depths, exponents, ls, marker=mk, color=color,
                    markersize=4.5,
                    label=rf"$k={k}$, $n={n}$")
    ax.axhline(3, color="#bbbbbb", lw=1)
    ax.axhline(6, color="#bbbbbb", lw=1)
    ax.text(11.6, 3.08, "pair-dominated envelope, $k=3$", color="#888888",
            fontsize=7.5, ha="right")
    ax.text(11.6, 5.62, "pair-dominated envelope, $k=4$", color="#888888",
            fontsize=7.5, ha="right")
    ax.set_xlabel(r"random depth $\tau_r$")
    ax.set_ylabel(r"$\ln m_k / \ln m_2$")
    ax.set_ylim(2, 6.4)
    # Legend in the empty band between the k = 3 curves (y <= 2.9) and the
    # k = 4 curves (y >= 3.9); lower right collides with the k = 3 series.
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left",
              bbox_to_anchor=(0.03, 0.24), handlelength=1.7,
              columnspacing=0.9)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_moments.pdf")
    plt.close(fig)


def require_ensembles() -> None:
    """Fail with the download instructions rather than deep inside a fit."""
    if list((ROOT / "results/pq").glob("pq_n*_sigma0.1.npz")):
        return
    raise SystemExit(
        "No restart ensembles found under results/.\n"
        "The figures are drawn from data archived separately at\n"
        "  https://doi.org/10.5281/zenodo.21710636\n"
        f"Unpack the archive so that results/ sits at {ROOT}.\n"
        "See DATA.md for the layout and the field-by-field schema."
    )


def main() -> None:
    require_ensembles()
    fig_scaling()
    fig_pq()
    fig_budget()
    fig_corrugation()
    fig_moments()
    for name in sorted(OUT.glob("*.pdf")):
        print("wrote", name.relative_to(ROOT))


if __name__ == "__main__":
    main()
