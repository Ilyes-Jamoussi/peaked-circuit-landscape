# The optimization landscape of peaked-circuit generation

Code, logs and figures for the manuscript *The optimization landscape of
peaked-circuit generation* (Ilyes Jamoussi, Polytechnique Montreal).

The study measures the variational landscape behind the peaked-circuit
construction of Aaronson and Zhang ([arXiv:2404.14493](https://arxiv.org/abs/2404.14493)):
the peakedness reachable by gradient descent, the geometry of the solutions it
finds, and how both degrade with system size.

Both archives are cited by their concept DOI, which always resolves to the
latest version; the manuscript cites the exact versions it was written
against.

- Manuscript: [arXiv:2608.11890](https://arxiv.org/abs/2608.11890)
- Code and logs: <https://doi.org/10.5281/zenodo.21810211>
- Restart ensembles, about 350 MB: <https://doi.org/10.5281/zenodo.21810321>

## Quick start

The reproduction repository is needed as a sibling directory; see
[requirements.txt](requirements.txt) for the exact commit the campaign ran
against.

```
git clone https://github.com/Ilyes-Jamoussi/peaked-circuits-pennylane.git
pip install -r requirements.txt
python test_pq.py                                    # 12 self-tests
python pq_experiment.py --num-qubits 8 --instance 0  # one solution ensemble
```

Regenerating the six manuscript figures additionally needs the restart
ensembles from the data archive, unpacked so that `results/` sits at the
repository root:

```
python figures/make_figures.py
```

## Where to look

[analysis/verify_claims.py](analysis/verify_claims.py) recomputes 311 of the
manuscript's quoted numbers from the archives and refuses to pass on any
disagreement, on a short corpus, or on a missing section; the registered
predictions quote the author's working record, as the manuscript's Appendix B
declares. Forty
of the fifty-one logs open on a line of self-tests against closed forms,
Haar limits or Monte-Carlo cross-checks; those scripts refuse to report if the
checks fail.

[REGISTRATION.md](REGISTRATION.md) fixes the statistic, the test and the verdict
rule for the corrugation sweep, and
[REGISTRATION-CONVERGED.md](REGISTRATION-CONVERGED.md) does the same for the
converged campaign and the optimizer-class block; both were committed before
the runs they govern.

| To check | Read |
|---|---|
| how peakedness is defined, optimized and stored | [pq_experiment.py](pq_experiment.py) |
| how near-optimal solutions and normalized overlaps are defined | [pq_analysis.py](pq_analysis.py) |
| that the campaign is complete and passes its integrity gate | [analysis/campaign_verdicts.log](analysis/campaign_verdicts.log) |
| whether restarts converged, and the instance ceiling | [analysis/budget_scan.log](analysis/budget_scan.log), [analysis/delta_star_curve.log](analysis/delta_star_curve.log) |
| the first-rung identity against the geometric measure of entanglement | [analysis/geometric_measure.log](analysis/geometric_measure.log) |
| that the first-rung equality holds by direct inversion, at every ground-truth point | [analysis/firstrung_inversion.log](analysis/firstrung_inversion.log) |
| the exact third and fourth moments by signed spin transfer | [analysis/third_moment.log](analysis/third_moment.log), [analysis/fourth_moment.log](analysis/fourth_moment.log) |
| the finite-depth covariance kernel against Monte Carlo | [analysis/kernel_robust.log](analysis/kernel_robust.log) |
| paths between pairs of reachable solutions | [analysis/connectivity_hires_pairs.log](analysis/connectivity_hires_pairs.log) |
| gradient, Hessian and shelf dimension at solutions | [analysis/hessian_solutions.log](analysis/hessian_solutions.log) |
| that the gauge dimension is the derived count 3S, exhibited and saturated | [analysis/gauge_dimension.log](analysis/gauge_dimension.log) |
| that the flat band at the solutions is the objective's own gauge, 3S + 9B + 4E | [analysis/probe_gauge.log](analysis/probe_gauge.log) |
| the entanglement-truncation ceiling | [analysis/ceiling_bound.log](analysis/ceiling_bound.log) |
| the exact kernel against Monte Carlo, 25 points with bootstrap errors | [analysis/kernel_exact.log](analysis/kernel_exact.log) |
| the 240-point residuals against Eq. (5), not against the deep limit | [analysis/check_kernel_exact_residuals.py](analysis/check_kernel_exact_residuals.py) |
| that the fifteen-rotation gate word really covers SU(4) | [analysis/gate_surjectivity.log](analysis/gate_surjectivity.log) |
| how much of the moment excess is the probe point | [analysis/moment_probe_n8_m2.log](analysis/moment_probe_n8_m2.log), [analysis/moment_probe_n6_m2.log](analysis/moment_probe_n6_m2.log) |
| that the two optimizers see the same gradients and pace differently | [analysis/optimizer_pacing.log](analysis/optimizer_pacing.log) |
| whether the 400-step cap, not the landscape, sets the reach | [analysis/step_budget_control.py](analysis/step_budget_control.py) |
| the reach law at optimizer convergence, both grids on one page | [analysis/converged_reach.log](analysis/converged_reach.log) |
| that the converged numbers count as converged, and the registered deficit test | [analysis/convergence_diagnostics.log](analysis/convergence_diagnostics.log) |
| whether another in-class optimizer reaches further, and what the exceedance is worth | [analysis/optclass_reach.log](analysis/optclass_reach.log) |
| **that the manuscript's numbers still match the archives** | [analysis/verify_claims.py](analysis/verify_claims.py) |
| how the campaign was actually run | [cloud/runner.py](cloud/runner.py) |

Each `analysis/*.py` carries its own usage line and runtime in its docstring.

## Layout

```
analysis/    34 scripts and their committed logs
figures/     make_figures.py and the six manuscript figures
cloud/       campaign runner and VM bootstrap
results/     restart ensembles (not in git, see DATA.md) and MANIFEST.sha256
```

The manifest is the one part of `results/` that is in git: it pins every
archive by SHA-256, so a download from the data archive can be checked against
a public, timestamped record with `python pq_validate.py`.

## Data

The restart ensembles are archived separately. [DATA.md](DATA.md) documents
every field of every archive, the file-naming scheme, and the overlap
conventions used throughout.

## Reproducing the campaign

Full regeneration of the ensembles is days of CPU time. [cloud/](cloud/) holds
what produced them: `bootstrap.sh` prepares a fresh Linux box, and `runner.py`
drives the job list, skipping any block whose output already exists and running
the integrity gate after each one.

```
python cloud/runner.py --list        # the job table
python cloud/runner.py --mini        # miniature end-to-end dry run
```

## Related repository

[peaked-circuits-pennylane](https://github.com/Ilyes-Jamoussi/peaked-circuits-pennylane):
the reproduction of [arXiv:2404.14493](https://arxiv.org/abs/2404.14493)
Section 3, which this work builds on.

## License

MIT, see [LICENSE](LICENSE).
