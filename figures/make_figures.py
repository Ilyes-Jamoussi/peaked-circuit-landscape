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


def fig_intro() -> None:
    """The object of study: circuit geometry and what a peaked output is.

    Top panel: the Sec. 2 geometry at the operating point, drawn exactly at
    (n, tau_r, tau_p) = (10, 10, 5). Bottom panels: the exact output
    distribution of the converged n = 10, instance 0 ensemble, regenerated
    from the archived angles and the instance seed; the regenerated
    delta(theta_best) and delta(0) match the archived scalars to 1e-14.
    """
    path = ROOT / "results/converged/pq_n10_i0_sigma0.1.npz"
    if not path.exists():
        print("  fig_intro: skipped, needs the converged archive "
              "results/converged/pq_n10_i0_sigma0.1.npz")
        return
    # The circuit builders live with the experiment code; pennylane is
    # imported here, not at module level, so the data-only figures do not
    # pay for it.
    from pq_experiment import (
        CircuitConfig, build_state_fn, sample_haar_random_layers,
    )
    from peaked_circuits import brick_wall_pairs

    data = np.load(path)
    n = int(data["num_qubits"])
    tau_r = int(data["num_random_layers"])
    tau_p = int(data["num_peaking_layers"])
    config = CircuitConfig(num_qubits=n, num_random_layers=tau_r,
                           num_peaking_layers=tau_p)
    rng = np.random.default_rng(np.random.SeedSequence(
        int(data["base_seed"]), spawn_key=(n, int(data["instance_index"]))
    ))
    state_fn = build_state_fn(config, sample_haar_random_layers(config, rng))
    best = int(np.argmax(data["peak_weights"]))
    probs_best = np.abs(np.asarray(state_fn(data["thetas_final"][best]))) ** 2
    probs_zero = np.abs(
        np.asarray(state_fn(np.zeros_like(data["thetas_final"][best])))
    ) ** 2
    for regenerated, archived in (
        (probs_best[0], float(data["peak_weights"][best])),
        (probs_zero[0], float(data["baseline_peak_weight"])),
    ):
        if abs(regenerated - archived) > 1e-12:
            raise SystemExit(
                f"fig_intro: regenerated value {regenerated!r} does not "
                f"match the archived scalar {archived!r}"
            )

    fig = plt.figure(figsize=(6.9, 4.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0],
                            hspace=0.34, wspace=0.10,
                            left=0.10, right=0.985, top=0.99, bottom=0.105)
    HAAR_FILL, HAAR_EDGE = "#d8d8d8", "#555555"
    PEAK_FILL, PEAK_EDGE = "#F7D488", "#B97E00"

    # --- Top: the brickwall at (n, tau_r, tau_p) = (10, 10, 5). ---
    ax = fig.add_subplot(grid[0, :])
    ax.set_xlim(-1.3, 28.0)
    ax.set_ylim(-2.9, 11.0)
    ax.axis("off")
    wire_y = {q: float(n - q) for q in range(n)}
    for q in range(n):
        ax.plot([-0.15, 15.55], [wire_y[q]] * 2, color="#888888", lw=0.7,
                zorder=1)
        ax.text(-0.45, wire_y[q], r"$|0\rangle$", ha="right", va="center",
                fontsize=7)
        ax.text(15.85, wire_y[q], r"$\langle 0|$", ha="left", va="center",
                fontsize=7)
    exploded = (14, 4)  # (layer index, top qubit of the brick blown up)
    for layer in range(tau_r + tau_p):
        x = 0.5 + layer
        trainable = layer >= tau_r
        for q, _ in brick_wall_pairs(layer, n):
            top, bottom = wire_y[q] + 0.33, wire_y[q + 1] - 0.33
            emphasized = (layer, q) == exploded
            ax.add_patch(plt.Rectangle(
                (x - 0.38, bottom), 0.76, top - bottom,
                facecolor=PEAK_FILL if trainable else HAAR_FILL,
                edgecolor=PEAK_EDGE if trainable else HAAR_EDGE,
                lw=1.4 if emphasized else 0.7, zorder=2,
            ))
    for x0, x1, label in (
        (0.12, 9.88, r"$U_r$: $\tau_r = n$ layers, i.i.d. Haar on $U(4)$"),
        (10.12, 14.88, r"$V(\theta)$: $\tau_p = n/2$ layers"),
    ):
        ax.plot([x0, x0, x1, x1], [0.42, 0.18, 0.18, 0.42],
                color="#444444", lw=0.7)
        ax.text((x0 + x1) / 2, -0.42, label, ha="center", va="top",
                fontsize=7.5)

    # The exploded trainable brick: the ordered fifteen-rotation word.
    ex_x = 0.5 + exploded[0]
    ex_top, ex_bot = wire_y[exploded[1]] + 0.33, wire_y[exploded[1] + 1] - 0.33
    dy1, dy2 = 7.05, 4.45
    for src, dst in (((ex_x + 0.38, ex_top), (17.8, dy1)),
                     ((ex_x + 0.38, ex_bot), (17.8, dy2))):
        ax.plot([src[0], dst[0]], [src[1], dst[1]], ls=(0, (2, 2)),
                color=PEAK_EDGE, lw=0.7, zorder=1)
    for y in (dy1 - 0.72, dy2 + 0.72):
        ax.plot([17.8, 27.0], [y] * 2, color="#888888", lw=0.7, zorder=1)
    word_boxes = ((18.7, r"$R_{XI}$", r"$\theta_1$"),
                  (20.4, r"$R_{YI}$", r"$\theta_2$"),
                  (22.1, r"$R_{ZI}$", r"$\theta_3$"),
                  (26.0, r"$R_{IZ}$", r"$\theta_{15}$"))
    for cx, word, angle in word_boxes:
        ax.add_patch(plt.Rectangle(
            (cx - 0.72, dy2 + 0.10), 1.44, dy1 - dy2 - 0.20,
            facecolor=PEAK_FILL, edgecolor=PEAK_EDGE, lw=0.7, zorder=2))
        ax.text(cx, (dy1 + dy2) / 2 + 0.28, word, ha="center", va="center",
                fontsize=7)
        ax.text(cx, (dy1 + dy2) / 2 - 0.55, angle, ha="center", va="center",
                fontsize=6)
    ax.text(24.05, (dy1 + dy2) / 2, r"$\cdots$", ha="center", va="center",
            fontsize=9)
    ax.text(22.4, dy1 + 1.95, "one trainable gate:",
            ha="center", va="bottom", fontsize=7)
    ax.text(22.4, dy1 + 0.95, "the ordered 15-rotation Pauli word",
            ha="center", va="bottom", fontsize=7)
    ax.text(22.4, dy2 - 1.15,
            r"$R_Q(\theta) = e^{-i\theta Q/2}$;"
            r"  $V(0) = \mathrm{identity}$",
            ha="center", va="top", fontsize=7.5)
    ax.text(22.4, 0.0,
            r"$\delta(\theta) = |\langle 0^n|\,V(\theta)\,U_r\,"
            r"|0^n\rangle|^2$",
            ha="center", va="center", fontsize=9)
    ax.text(22.4, -1.35, r"$(\tau_r, \tau_p) = (n, n/2)$, drawn at $n = 10$",
            ha="center", va="center", fontsize=7.5)

    # --- Bottom: the exact output distribution, before and after. ---
    axes = [fig.add_subplot(grid[1, 0])]
    axes.append(fig.add_subplot(grid[1, 1], sharey=axes[0]))
    floor = 2.0 ** (-n)
    for axis, probs, title in (
        (axes[0], probs_zero,
         r"$\theta = 0$: the scrambled state $U_r|0^n\rangle$"),
        (axes[1], probs_best,
         r"$\theta_{\mathrm{best}}$: best archived restart"),
    ):
        axis.plot(np.arange(2 ** n), probs, ".", ms=2.0, color="#9a9a9a",
                  zorder=2)
        axis.plot(0, probs[0], "o", ms=5, mfc=PEAK_FILL, mec=PEAK_EDGE,
                  mew=1.1, zorder=3)
        axis.axhline(floor, ls="--", lw=0.8, color="#555555", zorder=1)
        axis.set_yscale("log")
        axis.set_ylim(1e-7, 3.0)
        axis.set_xlim(-22, 2 ** n + 21)
        axis.set_xticks((0, 256, 512, 768, 1023))
        axis.set_title(title)
        axis.set_xlabel(r"basis state $x$")
    axes[0].annotate(rf"$\delta(0) = {probs_zero[0]:.1e}$".replace(
        "e-03", r" \times 10^{-3}"),
        xy=(0, probs_zero[0]), xytext=(150, 0.15),
        fontsize=8, ha="left",
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7,
                        shrinkB=4))
    axes[1].annotate(rf"$\delta = {probs_best[0]:.3f}$",
                     xy=(0, probs_best[0]), xytext=(150, 0.3),
                     fontsize=8, ha="left",
                     arrowprops=dict(arrowstyle="-", color="#666666",
                                     lw=0.7, shrinkB=4))
    axes[0].annotate(r"Haar floor $2^{-n}$",
                     xy=(480, floor), xytext=(640, 1.1e-6),
                     fontsize=8, ha="center", color="#555555",
                     arrowprops=dict(arrowstyle="-", color="#666666",
                                     lw=0.7, shrinkB=3))
    axes[0].set_ylabel(r"$|\langle x|\,V(\theta)\,U_r\,|0^n\rangle|^2$")
    plt.setp(axes[1].get_yticklabels(), visible=False)

    fig.savefig(OUT / "fig_intro.pdf")
    plt.close(fig)


def load_bests() -> dict[int, np.ndarray]:
    bests: dict[int, list[float]] = {}
    for name in sorted(glob.glob(str(ROOT / "results/pq/pq_n*_sigma0.1.npz"))):
        match = re.search(r"pq_n(\d+)_i(\d+)_", name)
        values = np.load(name)["peak_weights"]
        if len(values) < 200:
            continue
        bests.setdefault(int(match.group(1)), []).append(float(values.max()))
    return {n: np.array(v) for n, v in bests.items()}


def load_converged_r16() -> dict[int, np.ndarray]:
    """Per-instance E[max of 16] over the converged grid (B = 32)."""
    reach: dict[int, list[float]] = {}
    for name in sorted(glob.glob(
            str(ROOT / "results/converged/pq_n*_sigma0.1.npz"))):
        match = re.search(r"pq_n(\d+)_i(\d+)_", name)
        values = np.asarray(np.load(name)["peak_weights"], dtype=float)
        reach.setdefault(int(match.group(1)), []).append(
            float(expected_max(values, 16)))
    return {n: np.array(v) for n, v in reach.items()}


#: The five instance-averaged points of Ref. [aaronson2024peaked] Fig. 3c,
#: tau_p = tau_r/2 series (the operating regime). Their figure is published
#: as a raster with no tabulated values, so the points are digitized from
#: the panel: axes calibrated on the tick marks (five x ticks at n = 8..16,
#: log-y major ticks at one decade per 210 px), each data point read at the
#: center of its marker's bounding box by a triangle-template fit of the
#: series color. The quoted reading uncertainty is the half-height of the
#: marker in data units (about 12%), which dominates the one-pixel
#: calibration error. Digitization script committed alongside; their fitted
#: law evaluates to 1.189 per qubit and the digitized points give 1.20 over
#: n = 8..16, the agreement being the self-check.
AZ_FIG3C = {
    8: (0.728, 0.09),
    10: (0.512, 0.06),
    12: (0.416, 0.05),
    14: (0.238, 0.03),
    16: (0.169, 0.02),
}


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
    # Quoted on ln(mean), the quantity fitted, as everywhere in the text.
    shortfall = ((np.log(extrapolated) - np.log(mean[-1]))
                 / (sem[-1] / mean[-1]))
    ax.text(15.75, np.sqrt(mean[-1] * extrapolated),
            rf"${shortfall:.1f}\sigma$", ha="right", va="center")

    # The converged grid at the matched B0 = 16 estimator: the level with
    # the step cap removed, drawn as open markers beside the frozen means.
    converged = load_converged_r16()
    if converged:
        conv_sizes = np.array(sorted(converged))
        conv_mean = np.array([converged[n].mean() for n in conv_sizes])
        conv_sem = np.array([converged[n].std(ddof=1)
                             / np.sqrt(len(converged[n]))
                             for n in conv_sizes])
        ax.errorbar(conv_sizes + 0.18, conv_mean, yerr=conv_sem, fmt="s",
                    markerfacecolor="none", markeredgecolor="#0072B2",
                    ecolor="#0072B2", elinewidth=0.8, markersize=4.5,
                    capsize=2, zorder=4,
                    label=r"converged, $R_{16}$")

    # The five instance-averaged points of Ref. [aaronson2024peaked]
    # Fig. 3c, digitized (see AZ_FIG3C); error bars are the reading
    # uncertainty, not their statistical error, which the raster does not
    # resolve.
    az_sizes = np.array(sorted(AZ_FIG3C))
    az_vals = np.array([AZ_FIG3C[n][0] for n in az_sizes])
    az_err = np.array([AZ_FIG3C[n][1] for n in az_sizes])
    ax.errorbar(az_sizes - 0.18, az_vals, yerr=az_err, fmt="^",
                markerfacecolor="none", markeredgecolor="#D55E00",
                ecolor="#D55E00", elinewidth=0.8, markersize=5, capsize=2,
                zorder=4, label="Ref. [AZ] Fig. 3c (digitized)")

    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlabel(r"qubits $n$")
    ax.set_ylabel(r"mean-of-best $\overline{\delta}_{\mathrm{best}}$")
    ax.legend(frameon=False, loc="lower left", handlelength=1.6,
              fontsize=7)
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
    """Corrugation depth vs n, matched resolution, three instances per size.

    The statistic and the error bar are the ones registered in
    REGISTRATION.md: median over the twelve pairs of an instance, mean over
    instances, uncertainty the standard error over instances. The pair
    bootstrap the first pass used carried no instance-to-instance
    component, which is the variance that dominates here.
    """
    instances = (0, 1, 2)
    sizes, means, errors = [], [], []
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for n in SIZES:
        per_instance = []
        for instance in instances:
            tag = f"_i{instance}" if instance else ""
            path = ROOT / f"results/connectivity/conn_n{n}{tag}_o150s1_m64.npz"
            if not path.exists():
                continue
            data = np.load(path)
            ratios = np.array([
                data[key][2] / min(data[key][0], data[key][1])
                for key in data.keys()
                if key.startswith("pair") and key.endswith("_scalars")
            ])
            per_instance.append(float(np.median(ratios)))
            ax.plot([n] * len(ratios), ratios, ".", color=COLOR[n],
                    alpha=0.30, markersize=3)
        if len(per_instance) < 2:
            continue
        sizes.append(n)
        means.append(float(np.mean(per_instance)))
        errors.append(float(np.std(per_instance, ddof=1)
                            / np.sqrt(len(per_instance))))
        ax.plot([n] * len(per_instance), per_instance, "_", color=COLOR[n],
                markersize=9, markeredgewidth=1.4, zorder=3)
    if len(sizes) < 3:
        plt.close(fig)
        print("  fig_corrugation: skipped, the matched-resolution sweep needs "
              f"at least two instances at three sizes (have {len(sizes)}); "
              "run the connectivity_m64 block of cloud/runner.py")
        return
    sizes = np.array(sizes, dtype=float)
    means = np.array(means)
    errors = np.array(errors)
    ax.errorbar(sizes, means, yerr=errors, fmt="none", ecolor="#333333",
                elinewidth=1.1, capsize=3, zorder=4)
    for n, value in zip(sizes, means):
        ax.plot(n, value, MARKER[int(n)], color=COLOR[int(n)], markersize=6,
                zorder=5)

    # Weighted by the instance standard error, as registered.
    weights = means / errors
    exp_fit = np.polyfit(sizes, np.log(means), 1, w=weights)
    pow_fit = np.polyfit(np.log(sizes), np.log(means), 1, w=weights)
    grid = np.linspace(sizes.min() - 0.5, sizes.max() + 0.5, 60)
    ax.plot(grid, np.exp(np.polyval(exp_fit, grid)), "-", color="#444444",
            lw=1, label=rf"$\varrho \propto {np.exp(exp_fit[0]):.3f}^{{\,n}}$")
    ax.plot(grid, np.exp(np.polyval(pow_fit, np.log(grid))), "--",
            color="#999999", lw=1,
            label=rf"$\varrho \propto n^{{{pow_fit[0]:.1f}}}$")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xlabel(r"qubits $n$")
    ax.set_ylabel(r"corrugation floor $\varrho$")
    ax.legend(frameon=False, fontsize=8, loc="lower left", handlelength=1.6)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_corrugation.pdf")
    plt.close(fig)


TRUNCATION_TOLERANCE = 0.01   # the bound the caption asserts
POLE_MARGIN = 0.05            # |ln m2| below this makes the exponent unstable


def read_moment_scan(path: Path, num_qubits: int) -> dict[int, tuple]:
    """{tau_r: (m2, m3, m4)} from a committed fourth-moment scan log.

    Rows are dropped, with the reason printed, when the certified
    truncation bound exceeds TRUNCATION_TOLERANCE of m4 (the figure
    caption asserts 1%), or when m2 sits so close to 1 that the exponent
    ln m_k / ln m_2 is near its pole.
    """
    dimension = 2.0**num_qubits
    rows: dict[int, tuple] = {}
    bounds: dict[int, float] = {}
    depth = None
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[0].isdigit():
            depth = int(fields[0])
            rows[depth] = tuple(float(fields[k]) for k in (1, 2, 3))
            # The `dropped` column is rounded to two significant figures;
            # it is a fallback only. Where the scan printed the certified
            # error in m4 units, that line is authoritative.
            bounds[depth] = float(fields[5]) * dimension**4 / 24.0
        elif depth is not None and "certified |m4 error|" in line:
            bounds[depth] = float(line.split("<=")[1].split()[0])

    table: dict[int, tuple] = {}
    for depth, (m2, m3, m4) in rows.items():
        if bounds[depth] > TRUNCATION_TOLERANCE * m4:
            print(f"  fig_moments: n={num_qubits} tau_r={depth} dropped, "
                  f"certified bound {bounds[depth]:.3g} is "
                  f"{bounds[depth] / m4:.1%} of m4")
            continue
        if abs(np.log(m2)) < POLE_MARGIN:
            print(f"  fig_moments: n={num_qubits} tau_r={depth} dropped, "
                  f"m2 = {m2:.3f} is too close to the exponent's pole")
            continue
        table[depth] = (m2, m3, m4)
    return table


def fig_moments() -> None:
    """Moment-hierarchy exponents vs depth, read from the committed logs.

    Values come from analysis/fourth_moment.log (n = 6) and
    analysis/fourth_moment_n8.log (n = 8), whose m3 columns are the
    dedicated S3 runs; the k = 3 exponents are ln m3 / ln m2 of those
    tables. Nothing is transcribed by hand, so the figure cannot drift
    from the logs it cites.
    """
    n6 = read_moment_scan(ROOT / "analysis/fourth_moment.log", 6)
    n8 = read_moment_scan(ROOT / "analysis/fourth_moment_n8.log", 8)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    # No legend: each curve is labelled in its own color at its endpoint,
    # and the k of each bundle is named by the envelope annotations.
    for data, n, color in ((n6, 6, "#56B4E9"), (n8, 8, COLOR[8])):
        depths = sorted(data)
        for k, ls, mk in ((3, "--", "s"), (4, "-", "o")):
            exponents = [np.log(data[t][k - 2]) / np.log(data[t][0])
                         for t in depths]
            ax.plot(depths, exponents, ls, marker=mk, color=color,
                    markersize=4.5)
            # The n = 6 curves end at tau_r = 6, under the n = 8 curve of
            # their bundle; their labels drop below the line to stay clear.
            offset = {(6, 3): (4, -9), (6, 4): (5, -7),
                      (8, 3): (5, 0), (8, 4): (5, 0)}[n, k]
            ax.annotate(rf"$n={n}$", (depths[-1], exponents[-1]),
                        textcoords="offset points", xytext=offset,
                        color=color, fontsize=7.5, va="center")
    ax.axhline(3, color="#bbbbbb", lw=1)
    ax.axhline(6, color="#bbbbbb", lw=1)
    ax.set_xlim(0.5, 9.7)
    ax.text(9.45, 3.08, "pair-dominated envelope, $k=3$", color="#888888",
            fontsize=7.5, ha="right")
    ax.text(9.45, 6.05, "pair-dominated envelope, $k=4$", color="#888888",
            fontsize=7.5, ha="right")
    ax.set_xlabel(r"random depth $\tau_r$")
    ax.set_ylabel(r"$\ln m_k / \ln m_2$")
    ax.set_ylim(2, 6.4)
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
        "  https://doi.org/10.5281/zenodo.21875423\n"
        f"Unpack the archive so that results/ sits at {ROOT}.\n"
        "See DATA.md for the layout and the field-by-field schema."
    )


def main() -> None:
    require_ensembles()
    builders = (fig_intro, fig_scaling, fig_pq, fig_budget, fig_corrugation,
                fig_moments)
    written, skipped = [], []
    for builder in builders:
        before = {path: path.stat().st_mtime for path in OUT.glob("*.pdf")}
        builder()
        after = {path: path.stat().st_mtime for path in OUT.glob("*.pdf")}
        fresh = [p for p, t in after.items() if before.get(p) != t]
        (written if fresh else skipped).append(
            (builder.__name__, fresh[0] if fresh else None)
        )
    for name, path in written:
        print(f"wrote {path.relative_to(ROOT)}")
    # A figure the manuscript includes but this script did not produce is a
    # stale artefact; say so rather than listing it as written.
    for name, _ in skipped:
        print(f"NOT WRITTEN by {name}: any committed PDF for it is stale")
    if skipped:
        raise SystemExit(
            f"{len(skipped)} figure(s) not produced; the committed PDFs for "
            "them do not correspond to this code and this data."
        )


if __name__ == "__main__":
    main()
