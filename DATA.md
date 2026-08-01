# Data schema

The restart ensembles are archived at <https://doi.org/10.5281/zenodo.21710636>
(about 200 MB, 128 `.npz` archives). They are not part of this repository.

To use them with the code here, unpack the archive so that its `results/`
directory sits at the repository root. Every script and `figures/make_figures.py`
resolve their inputs relative to that path.

## Inventory

| Directory | Archives | Content |
|---|---|---|
| `results/pq/` | 77 | Main campaign. 18 instances at each of n = 8, 10, 12, 14; 4 at n = 16; one n = 12 control at `init_scale = 1.0`. |
| `results/budget800/` | 6 | Restart budget extended to 800 at n = 10, 12, three instances each. |
| `results/ceiling_curve/` | 17 | Peaking-depth sweep at small n, one directory per (n, tau_p). |
| `results/depth_ceiling/` | 8 | Deep random sections, one directory per tau_r. |
| `results/shallow_peaking/` | 6 | Reduced peaking depth at n = 8, 12. |
| `results/robustness/` | 5 | Optimizer and initialization variants at n = 10, instance 0. |
| `results/connectivity/` | 23 | String-method paths between solution pairs, including the 15 archives of the registered matched-resolution sweep. |
| `results/step_budget/` | 13 | The frozen protocol at a 1600-step cap on the protocol's own restart seeds, to size what the 400-step cap costs. |

File names encode the run: `pq_n{n}_i{instance}_sigma{init_scale}.npz`, and
`conn_n{n}[_i{instance}][_x{budget_factor}][_o{outer}s{steps}][_m{segments}].npz`
for connectivity. Instance 0 carries no `_i` tag, so the archives committed
before the instance sweep keep their original names.

The registered sweep of [REGISTRATION.md](REGISTRATION.md) is the fifteen
`conn_n{8,10,12,14,16}[_i{1,2}]_o150s1_m64.npz`: five sizes, three instances
each, twelve disjoint pairs and four scrambled controls per instance, all at
one resolution of 64 segments. Together they are the 180 paths behind the
corrugation figure.

## Restart ensembles

Every archive outside `results/connectivity/` shares one schema. `R` is the
number of restarts (200 in the main campaign and in `robustness/`, 800 in
`budget800/`, 60 in the depth and shallow runs, 20 or 60 in `ceiling_curve/`
depending on the peaking depth), and `P` the number of variational parameters.

`results/MANIFEST.sha256` pins every archive by SHA-256. It lives in the git
repository while the archives themselves live in the data archive, so a
download can be checked against a public, timestamped record:

```
python pq_validate.py          # structure, then the manifest, on all 128
```

Per-restart arrays:

| Field | Shape | Meaning |
|---|---|---|
| `peak_weights` | `(R,)` | **delta**, the probability of the `0...0` bit string under the peaked circuit. This is the quantity the paper calls delta and optimizes. |
| `thetas_init` | `(R, P)` | Initial parameters, drawn i.i.d. normal with standard deviation `init_scale`. |
| `thetas_final` | `(R, P)` | Parameters at the end of the restart. |
| `overlap_matrix` | `(R, R)` | `q_ij = |<psi_i|psi_j>|^2` between the final **solution** states `psi_i = V(theta_i) U_r |0...0>`, not the probe states; unit diagonal. |
| `num_steps` | `(R,)` | Optimizer steps actually taken, at most `max_steps`. |
| `restart_seconds` | `(R,)` | Wall-clock seconds per restart. |
| `argmax_indices` | `(R,)` | Index of the most probable bit string. A restart peaks on the target when this is 0. |
| `argmax_weights` | `(R,)` | Probability of that most probable string; equals `peak_weights` when the restart peaks on the target. |

Scalar metadata (zero-dimensional arrays): `num_qubits`, `num_random_layers`
(tau_r), `num_peaking_layers` (tau_p), `instance_index`, `base_seed`,
`init_scale`, `max_steps`, `min_steps`, `learning_rate`, `decay_every`,
`decay_factor`, `convergence_tol`, `elapsed_seconds`, and
`baseline_peak_weight`, the target-string probability before any optimization.

`P` follows from the brickwall geometry: every gate is a universal two-qubit
`ArbitraryUnitary` carrying 15 parameters, and the peaking layers continue the
brickwall pattern of the random section. Layer `tau_r + j` couples `n/2` pairs
when its index is even and `n/2 - 1` when odd, boundaries open. At n = 10 with
tau_r = 10 and tau_p = 5 the layer indices are 10 to 14, giving
15 * (5+4+5+4+5) = 345.

## Connectivity archives

These store one group of arrays per pair, with prefix `pair{k}_` for genuine
pairs of near-optimal solutions of the same instance and `control{k}_` for
scrambled controls, which pair a solution of one instance with a solution of
another evaluated on the first instance's field.

| Field | Shape | Meaning |
|---|---|---|
| `{tag}_values` | `(G,)` | delta along the final re-optimized path, on a dense grid. |
| `{tag}_raw_values` | `(G,)` | delta along the straight parameter segment between the endpoints. |
| `{tag}_waypoints` | `(W, P)` | Interior waypoints of the converged string. |
| `{tag}_overlap_a` | `(W,)` | Overlap of each waypoint state with endpoint A. |
| `{tag}_overlap_b` | `(W,)` | Overlap of each waypoint state with endpoint B. |
| `{tag}_scalars` | `(7,)` | Summary, in order: `delta_a`, `delta_b`, `min_interior`, `raw_min_interior`, `initial_length`, `final_length`, `min_adjacent_overlap`. |

The corrugation depth reported in the paper is
`min_interior / min(delta_a, delta_b)`, read from `{tag}_scalars`.

## Conventions

Overlaps are squared magnitudes throughout, so they lie in `[0, 1]` and a pair
of identical states gives 1. The floor-normalized overlap used in the paper is
`q_hat = (q - delta_i * delta_j) / (1 - delta_i * delta_j)`, which removes the
agreement two states share merely by both being peaked on the same string;
`pq_analysis.pair_overlaps` computes it.

Instances are indexed from 0 and generated from `base_seed` (42 throughout the
campaign), so a given `(num_qubits, instance_index)` names the same circuit in
every run.
