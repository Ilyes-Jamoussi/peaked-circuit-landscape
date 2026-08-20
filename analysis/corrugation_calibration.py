"""Calibration of the corrugation functional-form test (post-registration).

The registered verdict compares an exponential and a power law for the
corrugation floor by weighted chi-square on three degrees of freedom,
with weights from the standard error over three instances. Treating
weights estimated on three samples as known makes the asymptotic
chi-square reference anti-conservative. This control calibrates the
registered statistic by simulation: draw three instance medians per
size from the fitted power law with the observed per-size scatter,
re-estimate the weights, refit, and read the exclusion p from the
simulated distribution of the same chi-square.

Reading: the asymptotic reference gives p = 0.002; calibrated at the
actual sample size the same statistic gives p of order 0.1, and the
pooled-weight variant agrees. A fixed exponent is not excluded at
p < 0.05, so the second condition of the registered verdict fails
under calibration and the registered fallback applies.

Data provenance: the per-instance medians are parsed from the
committed analysis/campaign_verdicts.log, the log of record for the
registered sweep.

Self-tests: the parsed medians reproduce the published means and
standard errors; the two fits reproduce the committed chi-squares
(3.36 and 14.38), the base 0.871 and the exponent -1.50.

Usage:
    python analysis/corrugation_calibration.py        # ~1 min
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

LOG = Path(__file__).resolve().parent / "campaign_verdicts.log"
DRAWS = 40_000
SEED = 7


def parse_medians():
    rows = {}
    pattern = re.compile(
        r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+), ([\d.]+), ([\d.]+)\s*$"
    )
    for line in LOG.read_text().splitlines():
        match = pattern.match(line)
        if match:
            n = int(match.group(1))
            rows[n] = [float(match.group(k)) for k in (4, 5, 6)]
    assert sorted(rows) == [8, 10, 12, 14, 16], sorted(rows)
    return rows


def wls_chi2(y, sigma, x):
    weights = 1.0 / sigma**2
    design = np.column_stack([np.ones_like(x), x])
    normal = design.T @ (weights[:, None] * design)
    beta = np.linalg.solve(normal, design.T @ (weights * y))
    residual = y - design @ beta
    return float(np.sum(weights * residual**2)), beta


def main():
    med = parse_medians()
    ns = np.array(sorted(med), dtype=float)
    mean = np.array([np.mean(med[n]) for n in sorted(med)])
    se = np.array([np.std(med[n], ddof=1) / np.sqrt(3) for n in sorted(med)])
    assert np.allclose(mean, [0.7258, 0.5603, 0.4435, 0.3100, 0.2258],
                       atol=5e-4), mean
    assert np.allclose(se, [0.0252, 0.0054, 0.0143, 0.0268, 0.0169],
                       atol=5e-4), se

    y = np.log(mean)
    sy = se / mean
    chi_exp, beta_exp = wls_chi2(y, sy, ns)
    chi_pow, beta_pow = wls_chi2(y, sy, np.log(ns))
    assert abs(chi_exp - 3.36) < 0.05, chi_exp
    assert abs(chi_pow - 14.38) < 0.10, chi_pow
    assert abs(np.exp(beta_exp[1]) - 0.871) < 0.001
    assert abs(beta_pow[1] + 1.50) < 0.01
    print("self-tests passed (parsed medians reproduce the published means "
          "and errors; chi2 3.36 and 14.38, base 0.871, exponent -1.50)")

    p_asym = chi2_dist.sf(chi_pow, 3)
    print(f"\nregistered statistic, asymptotic reference: "
          f"chi2 = {chi_pow:.2f} on 3 dof, p = {p_asym:.4f}")

    # Calibration: null = fitted power law, per-size scatter of the three
    # ln-medians re-estimated on every draw, exactly as the statistic does.
    rng = np.random.default_rng(SEED)
    sigma_true = np.array([np.std(np.log(med[n]), ddof=1)
                           for n in sorted(med)])
    x = np.log(ns)
    y_true = beta_pow[0] + beta_pow[1] * x
    exceed = exceed_pooled = 0
    for _ in range(DRAWS):
        draws = y_true[:, None] + sigma_true[:, None] * rng.standard_normal(
            (5, 3))
        y_sim = draws.mean(axis=1)
        se_sim = draws.std(axis=1, ddof=1) / np.sqrt(3)
        chi_sim, _ = wls_chi2(y_sim, se_sim, x)
        exceed += chi_sim >= chi_pow
        pooled = np.full(5, np.sqrt(np.mean(draws.var(axis=1, ddof=1)) / 3))
        chi_pool, _ = wls_chi2(y_sim, pooled, x)
        exceed_pooled += chi_pool >= chi_pow
    p_cal = exceed / DRAWS
    p_pool = exceed_pooled / DRAWS
    print(f"calibrated by simulation ({DRAWS} draws, weights re-estimated "
          f"per draw): p = {p_cal:.3f}")
    print(f"pooled-weight variant:                                        "
          f"p = {p_pool:.3f}")

    assert p_cal > 0.05 and p_pool > 0.05
    print("\na fixed exponent is not excluded at p < 0.05 under calibration:"
          "\nthe second condition of the registered verdict fails and the"
          "\nregistered fallback applies (the corrugation deepens with n;"
          "\nthese data do not determine the functional form)")


if __name__ == "__main__":
    main()
