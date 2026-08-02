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
    # Quoted on ln(mean), the quantity fitted, as everywhere in the text.
    shortfall = ((np.log(extrapolated) - np.log(mean[-1]))
                 / (sem[-1] / mean[-1]))
    ax.text(15.75, np.sqrt(mean[-1] * extrapolated),
            rf"${shortfall:.1f}\sigma$", ha="right", va="center")
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
        "  https://doi.org/10.5281/zenodo.21710636\n"
        f"Unpack the archive so that results/ sits at {ROOT}.\n"
        "See DATA.md for the layout and the field-by-field schema."
    )


def main() -> None:
    require_ensembles()
    builders = (fig_scaling, fig_pq, fig_budget, fig_corrugation, fig_moments)
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
