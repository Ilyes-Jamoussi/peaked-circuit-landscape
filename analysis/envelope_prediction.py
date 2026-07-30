"""The assembled reachability prediction.

Model: per cell, delta = s * e with ln s ~ N(-sigma^2/2, sigma^2)
(E[s] = 1) and e ~ Exp(mean 2^-n); reachable peakedness = the 1/N quantile
of the mixture, N = exp(c2 * P) effective cells, with the single constant
c2 calibrated on the independent deep-limit anchor
delta_PT(n=8) = 0.51 (depth-ceiling experiment, sigma = 0):
V_PT = 0.51 * 256 = c2 * P(8).

The quantile is solved with the exact mixture integral (no asymptotic
saddle: those predict delta > 1 at these sizes). Predictions are capped at
delta = 1 and the decay base is fitted on the unsaturated sizes.

Usage:
    python analysis/envelope_prediction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pq_experiment  # noqa: E402,F401
from peaked_circuits import CircuitConfig  # noqa: E402

#: per-instance best of 200 restarts, averaged over instances (First
#: Results) -- the paper's statistic; its fitted base is 1.195 +/- 0.011.
MEASURED_BEST = {8: 0.721, 10: 0.485, 12: 0.363, 14: 0.242}
SIGMA2 = {8: 0.164, 10: 0.157, 12: 0.136, 14: 0.130, 16: 0.125}  # 14+: extrapolated
#: mean over 60 restarts at tau_r = 32 (depth-ceiling run). The quantile
#: model describes a best-of-N, so the best (0.533) is the arguably
#: cleaner anchor; both refute identically (best anchor: c2 = 0.650,
#: prediction 1.00/1.00/1.00/0.42 -- checked 2026-07-16).
ANCHOR_DELTA_PT = 0.51
ANCHOR_N = 8


def parameters(num_qubits: int) -> int:
    return CircuitConfig(
        num_qubits, num_qubits, num_qubits // 2
    ).num_peaking_parameters


def mixture_log_tail(value: float, sigma2: float) -> float:
    """log P(sE > value), exact numerical integral over ln s."""
    sigma = np.sqrt(sigma2)
    grid = np.linspace(-8 * sigma, 8 * sigma, 4001) - sigma2 / 2.0
    log_weights = (
        -((grid + sigma2 / 2.0) ** 2) / (2.0 * sigma2)
        - value * np.exp(-grid)
    )
    reference = log_weights.max()
    integral = np.trapezoid(np.exp(log_weights - reference), grid)
    return float(
        reference
        + np.log(integral)
        - 0.5 * np.log(2.0 * np.pi * sigma2)
    )


def quantile(log_n: float, sigma2: float) -> float:
    """V such that N * P(sE > V) = 1."""
    low, high = 1.0, 1e9
    for _ in range(200):
        mid = np.sqrt(low * high)
        if mixture_log_tail(mid, sigma2) + log_n > 0:
            low = mid
        else:
            high = mid
    return float(np.sqrt(low * high))


def main() -> None:
    c2 = ANCHOR_DELTA_PT * 2.0**ANCHOR_N / parameters(ANCHOR_N)
    print(f"calibration: c2 = {c2:.4f} from the deep-limit anchor at n = {ANCHOR_N}")
    print(
        f"\n{'n':>4} {'P':>5} {'ln N':>7} {'V_pred':>9} {'delta_pred':>10} "
        f"{'capped':>7} {'measured':>9}"
    )
    predictions = {}
    for num_qubits in (8, 10, 12, 14, 16):
        dim = 2.0**num_qubits
        p = parameters(num_qubits)
        log_n = c2 * p
        value = quantile(log_n, SIGMA2[num_qubits])
        raw = value / dim
        capped = min(raw, 1.0)
        predictions[num_qubits] = capped
        measured = MEASURED_BEST.get(num_qubits)
        print(
            f"{num_qubits:>4} {p:>5} {log_n:>7.1f} {value:>9.1f} {raw:>10.3f} "
            f"{capped:>7.3f} "
            f"{measured if measured is not None else float('nan'):>9.3f}"
        )

    usable = [n for n in (10, 12, 14) if predictions[n] < 0.999]
    if len(usable) >= 2:
        slope = np.polyfit(
            usable, [np.log(predictions[n]) for n in usable], 1
        )[0]
        base_pred = float(np.exp(-slope))
        print(
            f"\npredicted decay base on n = {usable}: {base_pred:.3f} "
            f"(measured: 1.195 best-statistic fit, 1.220 ensemble-mean fit; "
            f"registered bands, set around the then-quoted 1.220: confirmed "
            f"[1.15, 1.30], refuted outside [1.05, 1.45])"
        )
    else:
        print(
            "\nprediction SATURATED (delta ~ 1) at all usable sizes: the "
            "log-normal tail extrapolated to the 1/N quantile overshoots — "
            "base effectively 1.0, outside [1.05, 1.45]: REFUTED as "
            "extrapolated; the true s-tail beyond the bulk must be thinner "
            "than log-normal."
        )

    # Robustness variants (05 section 5.4): uniform bulk sigma^2 in
    # place of the per-size calibration — the refutation must not hinge
    # on that convention.
    for sigma2 in (0.055, 0.075):
        variant = {}
        for num_qubits in (8, 10, 12, 14, 16):
            raw = quantile(c2 * parameters(num_qubits), sigma2) \
                / 2.0**num_qubits
            variant[num_qubits] = min(raw, 1.0)
        usable = [n for n in (10, 12, 14, 16) if variant[n] < 0.999]
        if len(usable) >= 2:
            slope = np.polyfit(
                usable, [np.log(variant[n]) for n in usable], 1
            )[0]
            print(f"variant sigma^2 = {sigma2:g}: saturated through "
                  f"n = {min(usable) - 2}, then falls with base "
                  f"{np.exp(-slope):.2f} on n = {usable}")
        else:
            print(f"variant sigma^2 = {sigma2:g}: saturated everywhere")


if __name__ == "__main__":
    main()
