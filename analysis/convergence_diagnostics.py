"""Are the published reach numbers converged, and by how much do they fall short?

The frozen protocol reports whatever the optimizer had reached at 400 Adam
steps. Convergence was never checked, only hoped for, and the one control that
tried to check it -- 1600 steps -- had not converged either. So the honest
question is not "did it converge" but "how much is still on the table, and can
that be bounded".

Four diagnostics, every one of them measured:

  1. Gain per budget doubling, g_k(n) = R(n, S_k) / R(n, S_(k-1)) - 1, read
     off the ladder each restart archives. This replaces "fraction of restarts
     at the cap", which says only that the optimizer was interrupted, with a
     number that says how much the interruption cost: "at the last doubling
     there was 0.15 % left".
  2. The registered Richardson band. REGISTRATION-CONVERGED.md fixes, before
     any converged run, the one-sided upward band that is published whenever
     the acceptance criterion fails: with L_k the ladder value at rung k,
     g = L_K - L_{K-1} and r = g / (L_{K-1} - L_{K-2}), the band is
     [L_K, L_K + g r / (1 - r)] for 0 < r < 1 and undefined otherwise. It is
     transcribed here verbatim, top three rungs and nothing else. Convergence
     is never proved. It is bounded, which is the strongest honest claim.
  3. A census of stopping causes per size, with the median stopping rung and
     the median gain contributed by the final polish, so a reader can see
     whether the ladder stopped runs or the budget ceiling did.
  4. The gradient paradox, which is a result in its own right. On this
     landscape, after a learning-rate decay the relative gradient collapses by
     more than an order of magnitude within a hundred steps while delta moves
     a few tenths of a percent -- and several percent is still available at
     longer budgets. A gradient that vanishes therefore does NOT certify
     convergence, which is exactly why the stopping criterion adopted for the
     converged protocol is the gain per doubling and not the gradient norm.

Diagnostics 1-3 need results/converged. Diagnostic 4 needs only the per-step
traces of analysis/optimizer_pacing.npz and the extended-budget ensembles of
results/step_budget, both already committed, so it runs today.

Self-tests: the ladder fill reproduces a known non-decreasing sequence from a
NaN-padded row, the Richardson band lands exactly on the analytic limit of a
geometric ladder and reports "undefined" rather than a number wherever the
registration says it is undefined, and the collapse statistic recovers a
planted factor from a synthetic trace.

Usage:
    python analysis/convergence_diagnostics.py   # seconds; converged grid optional
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

# Fixed timestamp so the regenerated PDF is byte-identical: matplotlib
# otherwise stamps the current date into every file (figures/make_figures.py).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")  # 2026-01-01 UTC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from budget_scan import expected_max  # noqa: E402

CONVERGED = ROOT / "results" / "converged"
EXTENDED = ROOT / "results" / "step_budget"
PACING = Path(__file__).resolve().parent / "optimizer_pacing.npz"
FIGURE = ROOT / "figures" / "fig_gradient_paradox.pdf"

CONVERGED_RESTARTS = 32
BASE_BUDGET = 16  # same primary budget as converged_reach.py

#: What pq_experiment._collect_converged writes for every converged restart,
#: and therefore what distinguishes a converged archive from a frozen one.
#: analysis/converged_reach.py gates the converged grid on the same tuple.
INSTRUMENTATION_KEYS = (
    "ladder_weights",
    "running_max",
    "stop_reasons",
    "legacy_weight",
    "legacy_stop_step",
)

#: REGISTRATION-CONVERGED.md, "When the numbers count as converged": the
#: relative gain of the last budget doubling (rung 12800 read against rung
#: 6400 at B0 = 16) must be at most this at every size. The Richardson band
#: of richardson_band() is what gets published wherever this fails.
ACCEPTANCE_REL_GAIN = 0.005

#: optimizer_pacing.py runs at the RestartSettings defaults, i.e. the frozen
#: protocol: 400 steps, stepsize halved every 300. Asserted on load, so a
#: regenerated trace file with other settings fails loudly instead of being
#: misread.
PACING_STEPS = 400
PACING_DECAY_EVERY = 300
PACING_SIZE = 10  # n of the traces, and of the step_budget ensembles compared
WINDOW = 30  # steps averaged on each side of the collapse measurement

COLOR_GRADIENT = "#0072B2"  # Okabe-Ito, as in figures/make_figures.py
COLOR_GAIN = "#D55E00"

RUN_CONVERGED = (
    "python pq_experiment.py --converged --restarts 32 --instance I "
    "--num-qubits N --output results/converged --restart-cache CACHE"
)


def load_converged(directory: Path) -> dict[int, list[dict]]:
    """Converged archives by size, keeping only the keys the diagnostics read."""
    grid: dict[int, list[dict]] = {}
    if not directory.exists():
        return grid
    for name in sorted(directory.glob("pq_n*_sigma0.1.npz")):
        blob = np.load(name)
        missing = [key for key in INSTRUMENTATION_KEYS if key not in blob.files]
        if missing:
            # A frozen-protocol archive dropped in the converged directory.
            # It is skipped rather than read, and named rather than dropped in
            # silence: a reader must not have to wonder which files were used.
            print(f"   skipped {name.name}: no converged instrumentation "
                  f"(missing '{missing[0]}'); it is not a converged archive")
            continue
        if len(blob["peak_weights"]) < CONVERGED_RESTARTS:
            continue
        num_qubits = int(re.search(r"pq_n(\d+)_i(\d+)_", name.name).group(1))
        grid.setdefault(num_qubits, []).append({
            "ladder": np.asarray(blob["ladder"], dtype=np.int64),
            "ladder_weights": np.asarray(blob["ladder_weights"], dtype=float),
            "peak_weights": np.asarray(blob["peak_weights"], dtype=float),
            "stop_reasons": np.asarray(blob["stop_reasons"]),
            "polish_gain": np.asarray(blob["polish_gain"], dtype=float),
            "legacy_weight": np.asarray(blob["legacy_weight"], dtype=float),
            "running_max": np.asarray(blob["running_max"], dtype=float),
        })
    return grid


def fill_ladder(ladder_weights: np.ndarray) -> np.ndarray:
    """Carry each restart's last archived value across the rungs it never ran.

    A NaN at rung k means the restart had already stopped, so its running max
    at rung k is the one it stopped with. Carrying it forward is what a
    campaign actually run at budget S_k would have reported; dropping the row
    instead would silently restrict every high rung to the restarts that kept
    going, which are the ones still improving.
    """
    filled = np.array(ladder_weights, dtype=float, copy=True)
    if filled.ndim != 2:
        raise ValueError("ladder_weights must be (restarts, rungs).")
    if np.isnan(filled[:, 0]).any():
        raise ValueError("a restart has no value at the first rung.")
    for column in range(1, filled.shape[1]):
        missing = np.isnan(filled[:, column])
        filled[missing, column] = filled[missing, column - 1]
    return filled


def ladder_reach(archives: list[dict], budget: int):
    """Reach at every rung: mean over instances, error over instances."""
    rungs = archives[0]["ladder"]
    for archive in archives[1:]:
        if not np.array_equal(archive["ladder"], rungs):
            raise ValueError("instances of one size disagree on the ladder.")
    per_instance = np.array([
        [expected_max(column, budget)
         for column in fill_ladder(archive["ladder_weights"]).T]
        for archive in archives
    ])
    mean = per_instance.mean(axis=0)
    sem = per_instance.std(axis=0, ddof=1) / np.sqrt(len(per_instance))
    return rungs, mean, sem


def doubling_gains(reach_per_rung: np.ndarray) -> np.ndarray:
    """g_k = R_k / R_(k-1) - 1, one value per rung after the first."""
    return reach_per_rung[1:] / reach_per_rung[:-1] - 1.0


def richardson_band(levels: np.ndarray) -> dict:
    """The registered one-sided upward band, transcribed to the letter.

    REGISTRATION-CONVERGED.md, "When the numbers count as converged", fixes it
    before any converged run:

        with L_k the ladder value at rung k, g = L_K - L_{K-1} and
        r = g / (L_{K-1} - L_{K-2}), the band is [L_K, L_K + g r / (1 - r)]
        for 0 < r < 1, and undefined otherwise, in which case the raw value
        and the last gain are reported alone.

    So r is the ratio of the last two ABSOLUTE ladder increments, read off the
    top three rungs and nothing else: not a decay rate fitted across the whole
    ladder, and not a ratio of relative gains. The distinction decides numbers,
    not presentation. On a ladder whose increments stall at the top -- exactly
    the regime where the acceptance criterion fails and this band is what gets
    published -- a decay rate fitted across all rungs is dominated by the early
    steep decay and can understate the band by an order of magnitude.

    Returns ``value`` = L_K and ``last_gain`` = g always, since the
    registration reports those two alone wherever the band is undefined, and
    ``r``, ``residual`` = g r / (1 - r) and ``top`` = L_K + residual, the last
    two in the units of L and None wherever the band is undefined.
    """
    levels = np.asarray(levels, dtype=float)
    absent = {"value": None, "last_gain": None, "r": None,
              "residual": None, "top": None}
    if levels.ndim != 1:
        raise ValueError("levels must be the one-dimensional ladder series.")
    if levels.size < 3:
        return absent | {"note": "fewer than three rungs; r needs two gains"}
    # fill_ladder carries every stopped restart forward and expected_max is
    # monotone in the sample, so the reach per rung cannot decrease. A decrease
    # means the caller did not build these levels that way, and r would then be
    # a ratio of quantities the registration never contemplated.
    slack = 1e-12 * float(np.max(np.abs(levels)))
    if np.any(np.diff(levels) < -slack):
        raise ValueError("ladder levels decrease; they must be the "
                         "forward-filled reach per rung, which is "
                         "non-decreasing by construction.")

    value = float(levels[-1])
    gain = float(levels[-1] - levels[-2])
    previous = float(levels[-2] - levels[-3])
    if previous == 0.0:
        return absent | {
            "value": value, "last_gain": gain,
            "note": "the previous gain is zero; r undefined, no band",
        }
    ratio = gain / previous
    if not 0.0 < ratio < 1.0:
        return absent | {
            "value": value, "last_gain": gain, "r": ratio,
            "note": "r outside (0, 1); band undefined, raw value and last "
                    "gain alone",
        }
    residual = gain * ratio / (1.0 - ratio)
    return {
        "value": value,
        "last_gain": gain,
        "r": ratio,
        "residual": residual,
        "top": value + residual,
        "note": "",
    }


def stop_census(archive: dict) -> dict:
    """Stopping causes, median stopping rung and median polish gain."""
    reasons = archive["stop_reasons"]
    filled = archive["ladder_weights"]
    reached = (~np.isnan(filled)).sum(axis=1) - 1
    return {
        "counts": {name: int((reasons == name).sum())
                   for name in ("cap", "rel_tol", "legacy_tol")},
        "stop_rung": float(np.median(archive["ladder"][reached])),
        "polish_gain": float(np.median(archive["polish_gain"])),
        "truncation_deficit": float(
            np.median(archive["peak_weights"] / archive["legacy_weight"] - 1.0)
        ),
    }


def relative_gradient(deltas: np.ndarray, gradient_norms: np.ndarray):
    """|grad| / delta: the fractional change in delta per unit of parameter."""
    return gradient_norms / deltas


def collapse_factor(trace: np.ndarray, decay_step: int, window: int) -> float:
    """Drop in the median relative gradient across the decay, end vs before.

    Medians over a window on each side, because the per-step gradient norm of
    an Adam run is far too noisy to be read at a single step.
    """
    before = np.median(trace[decay_step - window:decay_step])
    after = np.median(trace[-window:])
    return float(before / after)


def load_pacing() -> dict | None:
    """Adam traces of optimizer_pacing.npz, or None if it was never run."""
    if not PACING.exists():
        return None
    blob = np.load(PACING)
    restarts = sorted({int(match.group(1))
                       for key in blob.files
                       if (match := re.fullmatch(r"adam_r(\d+)_deltas", key))})
    if not restarts:
        return None
    deltas = np.array([blob[f"adam_r{r}_deltas"] for r in restarts])
    gradients = np.array([blob[f"adam_r{r}_gradnorm"] for r in restarts])
    if deltas.shape[1] != PACING_STEPS:
        raise ValueError(
            f"{PACING.name} holds {deltas.shape[1]}-step traces; this script "
            f"reads the frozen protocol's {PACING_STEPS}."
        )
    return {"deltas": deltas, "gradients": gradients,
            "decay_step": (PACING_STEPS - 1) // PACING_DECAY_EVERY
            * PACING_DECAY_EVERY}


def extended_budget_gains(num_qubits: int) -> dict | None:
    """What 400 -> 1600 steps still bought, on the campaign's own seeds."""
    paths = sorted(EXTENDED.glob(f"pq_n{num_qubits}_i*_sigma0.1.npz"))
    if not paths:
        return None
    best, mean = [], []
    for path in paths:
        long_run = np.load(path)
        instance = int(long_run["instance_index"])
        short = np.load(ROOT / "results" / "pq" /
                        f"pq_n{num_qubits}_i{instance}_sigma0.1.npz")
        count = len(long_run["peak_weights"])
        if not np.allclose(short["thetas_init"][:count], long_run["thetas_init"]):
            raise ValueError(f"n={num_qubits} i={instance}: the extended run "
                             "does not share the protocol's restart seeds")
        a, b = short["peak_weights"][:count], long_run["peak_weights"]
        best.append(float(b.max() / a.max() - 1.0))
        mean.append(float(b.mean() / a.mean() - 1.0))
    return {"instances": len(paths), "best": float(np.mean(best)),
            "best_max": float(np.max(best)), "mean": float(np.mean(mean))}


def write_figure(pacing: dict, extended: dict | None) -> Path | None:
    """Relative gradient and remaining gain, overlaid against the step.

    The within-trace remaining gain is measured against the 400-step value,
    so it collapses to zero at the right edge by construction. The horizontal
    line is what a longer budget actually bought at this size on the
    campaign's own seeds: the floor the dashed curve should be read against.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "figure.dpi": 150,
    })

    deltas, gradients = pacing["deltas"], pacing["gradients"]
    steps = np.arange(1, deltas.shape[1] + 1)
    ratio = relative_gradient(deltas, gradients)
    remaining = deltas[:, -1][:, None] / deltas - 1.0

    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    lo, mid, hi = np.percentile(ratio, [25, 50, 75], axis=0)
    ax.fill_between(steps, lo, hi, color=COLOR_GRADIENT, alpha=0.18, lw=0)
    ax.plot(steps, mid, color=COLOR_GRADIENT)
    ax.set_yscale("log")
    ax.set_xlabel("Adam step")
    ax.set_ylabel(r"relative gradient $|\nabla\delta| / \delta$",
                  color=COLOR_GRADIENT)
    ax.tick_params(axis="y", colors=COLOR_GRADIENT)
    ax.axvline(pacing["decay_step"], color="#888888", lw=0.8, ls=":")
    ax.annotate("stepsize halved", xy=(pacing["decay_step"], mid.max()),
                xytext=(-4, 0), textcoords="offset points", ha="right",
                va="top", fontsize=7, color="#555555", rotation=90)

    twin = ax.twinx()
    lo, mid, hi = np.percentile(remaining[:, :-1], [25, 50, 75], axis=0)
    twin.fill_between(steps[:-1], lo, hi, color=COLOR_GAIN, alpha=0.18, lw=0)
    twin.plot(steps[:-1], mid, color=COLOR_GAIN, ls="--",
              label=rf"$\delta_{{{PACING_STEPS}}}/\delta_s - 1$ (within trace)")
    twin.set_yscale("log")
    twin.set_ylim(1e-4, 10 * mid.max())
    twin.set_ylabel("gain still to come", color=COLOR_GAIN)
    twin.tick_params(axis="y", colors=COLOR_GAIN)
    twin.spines["top"].set_visible(False)
    if extended is not None:
        twin.axhline(extended["mean"], color=COLOR_GAIN, lw=1.0, ls="-",
                     alpha=0.75,
                     label=f"measured 400 -> 1600 steps "
                           f"({100 * extended['mean']:.1f}%)")
        twin.legend(fontsize=6.5, frameon=False, loc="lower left")

    ax.set_title(rf"$n = {PACING_SIZE}$, {len(deltas)} Adam restarts "
                 "(median, IQR band)")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)
    return FIGURE


def self_tests() -> None:
    padded = np.array([[0.10, 0.20, 0.25, np.nan, np.nan],
                       [0.10, 0.11, 0.12, 0.121, 0.1215]])
    filled = fill_ladder(padded)
    assert np.allclose(filled[0], [0.10, 0.20, 0.25, 0.25, 0.25]), filled[0]
    assert np.allclose(filled[1], padded[1]), filled[1]
    assert (np.diff(filled, axis=1) >= 0).all(), "fill must stay non-decreasing"

    # A geometric ladder whose limit is known in closed form. Take
    # L_k = limit - c rho^k: its increments are c rho^(k-1) (1 - rho), so
    # r = rho exactly, and the registered band's upper edge is
    #   L_K + g r / (1 - r) = limit - c rho^K + c rho^K = limit,
    # the analytic sum of the whole remaining tail. The band must land on it,
    # neither short (it would understate what the ladder left uncollected) nor
    # long (it would inflate a published upper bound).
    rho, limit, c = 0.4, 0.6, 0.05
    ladder = limit - c * rho ** np.arange(7)
    band = richardson_band(ladder)
    assert abs(band["r"] - rho) < 1e-12, band
    assert abs(band["top"] - limit) < 1e-14, band
    assert abs(band["residual"] - (limit - ladder[-1])) < 1e-14, band
    assert abs(band["residual"] - sum(band["last_gain"] * rho**k
                                      for k in range(1, 400))) < 1e-14, band
    # It is the top three rungs that decide it, and only those: prepending any
    # history the registration does not read must not move the band.
    padded = np.concatenate([[limit - c * 8.0, limit - c * 3.0], ladder])
    assert richardson_band(padded)["top"] == band["top"], padded

    # r >= 1: increments that stop shrinking. The registration calls the band
    # undefined there, so the function must report absence, not a number --
    # g r / (1 - r) would be negative or infinite and would read as a bound.
    # Dyadic levels, so the increments and their ratio are exact in binary and
    # the test measures the branch rather than the rounding.
    stalled = np.array([0.125, 0.1875, 0.3125, 0.5625])  # steps 1/16 1/8 1/4
    stalled_band = richardson_band(stalled)
    assert stalled_band["r"] == 2.0, stalled_band
    assert stalled_band["residual"] is None and stalled_band["top"] is None
    assert stalled_band["value"] == 0.5625 and stalled_band["last_gain"] == 0.25
    assert richardson_band(np.array([0.25, 0.5, 0.75]))["top"] is None  # r = 1

    # r <= 0: a ladder that stops moving. r = 0 is outside the registered
    # interval, so the band is undefined there too, not a zero-width band.
    assert richardson_band(np.array([0.10, 0.20, 0.20]))["residual"] is None
    # A zero previous increment would divide by zero; it must be caught first.
    assert richardson_band(np.array([0.20, 0.20, 0.21]))["residual"] is None
    assert richardson_band(np.array([0.20, 0.30]))["residual"] is None

    # A planted collapse must be recovered from a noisy trace.
    rng = np.random.default_rng(5)
    trace = np.concatenate([
        np.full(300, 1.0), np.full(100, 1.0 / 64.0)
    ]) * np.exp(0.05 * rng.standard_normal(400))
    measured = collapse_factor(trace, 300, WINDOW)
    assert abs(measured / 64.0 - 1.0) < 0.05, measured

    print(f"self-tests passed (ladder fill carries a stopped restart forward "
          f"and stays non-decreasing; the registered Richardson band lands on "
          f"the\nanalytic limit {limit} of a geometric ladder with r = {rho} "
          f"and is undefined for r <= 0 and r >= 1;\nplanted collapse of 64x "
          f"recovered as {measured:.0f}x)")


def report_converged(grid: dict[int, list[dict]]) -> None:
    sizes = sorted(grid)
    rungs = grid[sizes[0]][0]["ladder"]

    print("\n1. Gain per budget doubling\n")
    print(f"   g_k(n) = R(n, S_k) / R(n, S_(k-1)) - 1, with R the mean over "
          f"instances of\n   E[max of B = {BASE_BUDGET} restarts] at that "
          "rung. A restart that had already\n   stopped carries its value "
          "forward, which is what a campaign run at that\n   budget would "
          f"have reported. Columns are S_k; the first rung ({rungs[0]}) has "
          "no\n   predecessor and is omitted.\n")
    header = f"{'n':>4} {'inst':>5}"
    for rung in rungs[1:]:
        header += f" {rung:>9}"
    print(header)
    all_gains: dict[int, np.ndarray] = {}
    levels: dict[int, np.ndarray] = {}
    for num_qubits in sizes:
        _, mean, _ = ladder_reach(grid[num_qubits], BASE_BUDGET)
        gains = doubling_gains(mean)
        all_gains[num_qubits] = gains
        levels[num_qubits] = mean
        row = f"{num_qubits:>4} {len(grid[num_qubits]):>5}"
        for gain in gains:
            row += f" {100 * gain:>8.3f}%"
        print(row)

    print("\n2. What the ladder never collected: the registered Richardson "
          "band\n")
    print("   REGISTRATION-CONVERGED.md, verbatim: with L_k the ladder value "
          "at rung k,\n   g = L_K - L_{K-1} and r = g / (L_{K-1} - L_{K-2}), "
          "the band is\n   [L_K, L_K + g r / (1 - r)] for 0 < r < 1, and "
          "undefined otherwise, in which\n   case the raw value and the last "
          "gain are reported alone. It is ONE-SIDED\n   AND UPWARD: the "
          "converged reach can lie above L_K by up to the residual,\n   and "
          "nothing here bounds it from below. r is read off the top three "
          "rungs\n   only -- it is not a decay rate fitted across the ladder, "
          "which would be a\n   different and much narrower band on a ladder "
          "that stalls at the top.\n")
    print(f"   The band is what gets published wherever 'accept' reads FAIL, "
          f"i.e. wherever the\n   last doubling of this ladder "
          f"({rungs[-2]} -> {rungs[-1]}) still gains more than "
          f"{100 * ACCEPTANCE_REL_GAIN:.1f} %; the\n   registration fixes "
          "that acceptance at rung 12800 against rung 6400.\n")
    print(f"{'n':>4} {'L_K':>9} {'g':>10} {'g/L_K':>8} {'r':>7} "
          f"{'residual':>10} {'band top':>9} {'accept':>7}")
    for num_qubits in sizes:
        band = richardson_band(levels[num_qubits])
        last_relative = float(all_gains[num_qubits][-1])
        verdict = "pass" if last_relative <= ACCEPTANCE_REL_GAIN else "FAIL"
        if band["value"] is None:  # a ladder too short for r to exist
            print(f"{num_qubits:>4} {'-':>9} {'-':>10} "
                  f"{100 * last_relative:>7.3f}% {'-':>7} {'-':>10} {'-':>9} "
                  f"{verdict:>7}   {band['note']}")
            continue
        head = (f"{num_qubits:>4} {band['value']:>9.4f} "
                f"{band['last_gain']:>10.6f} {100 * last_relative:>7.3f}%")
        if band["residual"] is None:
            ratio = "-" if band["r"] is None else f"{band['r']:.3f}"
            print(f"{head} {ratio:>7} {'-':>10} {'-':>9} {verdict:>7}   "
                  f"{band['note']}")
            continue
        print(f"{head} {band['r']:>7.3f} {band['residual']:>10.6f} "
              f"{band['top']:>9.4f} {verdict:>7}")

    print("\n3. Why restarts stopped\n")
    print(f"{'n':>4} {'inst':>5} {'cap':>7} {'rel_tol':>8} {'legacy':>7} "
          f"{'median stop rung':>17} {'polish gain':>12} {'vs 400 steps':>13}")
    for num_qubits in sizes:
        counts = {"cap": 0, "rel_tol": 0, "legacy_tol": 0}
        rung, polish, deficit = [], [], []
        for archive in grid[num_qubits]:
            census = stop_census(archive)
            for name, value in census["counts"].items():
                counts[name] += value
            rung.append(census["stop_rung"])
            polish.append(census["polish_gain"])
            deficit.append(census["truncation_deficit"])
        total = sum(counts.values())
        print(f"{num_qubits:>4} {len(grid[num_qubits]):>5} "
              f"{counts['cap'] / total:>7.3f} {counts['rel_tol'] / total:>8.3f} "
              f"{counts['legacy_tol'] / total:>7.3f} "
              f"{int(np.median(rung)):>17} "
              f"{100 * np.median(polish):>11.3f}% "
              f"{100 * np.median(deficit):>12.2f}%")
    print("\n   'vs 400 steps' is the median per-restart deficit of the frozen "
          "stopping\n   rule, evaluated along the converged trajectory itself: "
          "no seed matching\n   and no architecture assumption enter it.")


def report_gradient_paradox(pacing: dict) -> dict | None:
    print("\n4. The gradient paradox: a small gradient does not certify "
          "convergence\n")
    deltas, gradients = pacing["deltas"], pacing["gradients"]
    decay = pacing["decay_step"]
    ratio = relative_gradient(deltas, gradients)
    factors = np.array([collapse_factor(row, decay, WINDOW) for row in ratio])
    motion = deltas[:, -1] / deltas[:, decay - 1] - 1.0
    extended = extended_budget_gains(PACING_SIZE)

    print(f"   Traces: analysis/optimizer_pacing.npz, n = {PACING_SIZE} "
          f"instance 0, {len(deltas)} Adam\n   restarts under the frozen "
          f"protocol; the stepsize is halved at step {decay}.\n")
    print(f"{'quantity':>44} {'median':>10} {'IQR':>19}")
    q1, q3 = np.percentile(factors, [25, 75])
    print(f"{'collapse of |grad|/delta, step ' + str(decay) + ' -> end':>44} "
          f"{np.median(factors):>9.0f}x {f'{q1:.0f}x - {q3:.0f}x':>19}")
    q1, q3 = np.percentile(100 * motion, [25, 75])
    print(f"{'delta gained over the same span':>44} "
          f"{100 * np.median(motion):>9.2f}% {f'{q1:.2f}% - {q3:.2f}%':>19}")

    if extended is None:
        print(f"\n   No extended-budget ensembles under {EXTENDED}, so what "
              "remains on the\n   table at this size cannot be quantified "
              "here; run the step_budget block.")
    else:
        print(f"\n   Still available at n = {PACING_SIZE} on the campaign's "
              f"own seeds, from\n   results/step_budget "
              f"({extended['instances']} instances, 400 -> 1600 steps): "
              f"{100 * extended['mean']:.2f}% on the\n   restart mean and "
              f"{100 * extended['best']:.2f}% on the instance best, up to "
              f"{100 * extended['best_max']:.2f}% on one\n   instance. The "
              "deficit grows with n; analysis/step_budget_control.py\n   "
              "prints it per size.")

    print("\n   So the relative gradient falls by more than an order of "
          "magnitude while\n   delta moves a few tenths of a percent, at a "
          "point where a longer budget\n   still buys percent-level gains. "
          "The gradient tracks the stepsize, not the\n   distance left to go: "
          "after a decay the iterate simply stops moving. This\n   is why the "
          "converged protocol stops on the gain per doubling and not on\n   "
          "a gradient norm, and why the 1600-step control stalled -- its "
          "stepsize\n   had decayed to nothing with no floor under it.")

    print("\n   What these traces cannot show: they stop at "
          f"{PACING_STEPS} steps, so the\n   'gain still to come' plotted "
          "against the step is measured against the\n   400-step value and is "
          "a LOWER BOUND on the true remaining gain. Traces\n   run under the "
          "converged protocol, with the learning-rate floor, would be\n   "
          "needed to draw the whole gap; they are not in this repository.")
    return extended


def main() -> None:
    self_tests()

    grid = load_converged(CONVERGED)
    if not grid:
        print(f"\nConverged grid not found under {CONVERGED}.")
        print(f"   Run, per size and instance:\n     {RUN_CONVERGED}")
        print("   Diagnostics 1-3 need it. Diagnostic 4 runs on the committed "
              "traces\n   and is reported below.")
    else:
        report_converged(grid)

    pacing = load_pacing()
    if pacing is None:
        print(f"\n4. Per-step traces not found under {PACING}.")
        print("   Run analysis/optimizer_pacing.py (~25 min), then rerun this.")
        return
    extended = report_gradient_paradox(pacing)

    figure = write_figure(pacing, extended)
    if figure is None:
        print("\n   matplotlib not available; the figure was not written.")
    else:
        print(f"\nwrote {figure}")


if __name__ == "__main__":
    main()
