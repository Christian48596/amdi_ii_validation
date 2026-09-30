# Reproducibility protocol

This document maps the revised manuscript claims to executable checks and archived artifacts.

## Level A — fast release verification

```bash
pytest -q
python verify_release.py
```

These commands check 22 unit/regression tests, checkpoint hashes, the frozen configuration, manuscript values, Section 5 diagnostics and controls, validation records, figure integrity, citation metadata, and the SHA-256 manifest.

## Level B — deterministic AMDI regression

```bash
python experiments/01_deterministic_backend_regression.py --config configs/publication.json
python experiments/02_protocol_alignment_audit.py --config configs/publication.json
python experiments/03_uniform_reference_check.py --config configs/publication.json
```

Acceptance criteria are:

- reconstructed-field relative error below `1e-12`;
- identical adaptive tree and coefficient discrepancy below `1e-12`;
- zero protocol tree distance and final difference within numerical precision;
- full reference basis size `1024` at `32×32`;
- monotone fixed-tree reference energy and accepted safeguards.

The backend implementation comparison and within-backend protocol comparison are separate checks. Their reported occupancies must not be interchanged.

## Level C — Section 5 action/energy audit

```bash
python experiments/11_action_energy_audit.py --device cpu --include-transfer
```

Outputs are written under `results/11_section5_diagnostics/`:

- `step_diagnostics.csv`: one row per case, method, and decision, including active degrees of freedom before/after adaptation, proposed and executed actions, `E_ref`, occupancy, switching, and energies before propagation, after propagation, and after adaptation;
- `deterministic_candidates.csv`: energy, tree penalty, total score, and selection status for unchanged, refinement, and coarsening candidates;
- `action_energy_by_step.csv`: nine-case action totals and counts of energy-increasing adaptations;
- `terminal_metrics.csv`, `summary.csv`, and `status.json`: case-level and aggregate checks.

The temporal control permits the actor to act at decision 1 and forces retention at decisions 2–6.

## Level D — validation-tuned controls

```bash
python experiments/12_selector_controls.py --device cpu --include-transfer
```

Outputs are written under `results/12_section5_controls/`. The retain-tree control never changes the threshold-initialized tree. Top-`K` and fixed-threshold rules act once at decision 1 and retain thereafter.

The control parameters are selected using validation images `100–103` with noise seed `20000 + image_seed`. Both the best-validation-return and closest-validation-occupancy choices are recorded in `selected_on_validation.json`; all choices are frozen before holdout testing.

The complete grids, nine-case paired differences, common four-image frontiers, and transfer results are archived. The threshold is applied to the raw norm of the three observed-image Haar detail coefficients. Parameters are not retuned at `64×64` or `128×128`.

## Level E — figure regeneration

```bash
python make_all_manuscript_figures.py --device cpu --outdir IMG_TEST
```

No training or parameter selection is performed.  The command reads the
archived Section 5 results and, by default, the archived selected-checkpoint
four-region panels.  Add `--recompute-four-region` to regenerate those panels
from `checkpoints/selected/best_policy.pt`.  The checked manuscript figures
are under `IMG/` as PDF and 600-dpi PNG pairs.

## Level F — checkpoint-only reevaluation

```bash
python reproduce_final_results.py --device cpu --outdir reproduced_results
```

This reevaluates the nine-case holdout, six PPO occupancy weights, resolution transfer, reward ablation, and three policy seeds. It does not retrain any policy.

## Exact datasets and noise

Clean-image seeds:

```text
training     0–15
validation  100–103
test        1000–1007
```

Gaussian-noise standard deviation is `0.08`; the noisy input is clipped to `[0,1]`.

| Evaluation | Noise seed |
|---|---|
| control tuning | `20000 + image_seed` |
| nine-case holdout | `30000 + image_seed` |
| four-region holdout | `137` |
| common PPO/control frontier subset | `40000 + image_seed` |
| resolution transfer | `50000 + image_seed + n` |
| reward ablation | `60000 + image_seed` |
| policy-seed robustness | `70000 + image_seed` |

Pareto/control frontiers, reward ablation, robustness, and transfer use the stated four-image subsets. The `lambda_occ = 0.15` PPO frontier point reuses the selected checkpoint.

## Metrics and action resolution

`E_ref` is terminal discrepancy from the uniformly resolved AMDI trajectory. `C_rel` is terminal active degrees of freedom divided by the full representation size. Switching is averaged over the six decisions.

Terminal reconstructions are clipped pointwise to `[0,1]` before RMSE and SSIM are evaluated against the clean image.

Local actions are `coarsen / retain / refine`. Coarsening executes only when all four active siblings vote to coarsen. Proposed actions and executed tree changes are recorded separately.

## VAMPyR/MRCPP diagnostic

VAMPyR/MRCPP is an order-five projection of the **clean analytic target**, using precision `10^-3` and maximum depth eight. It is neither the production AMDI propagator nor an independent validation of the learned trajectory. Its effective levels and Haar refinement levels are used only as qualitative, method-dependent localization indicators.
