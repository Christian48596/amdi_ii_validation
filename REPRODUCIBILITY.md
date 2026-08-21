# Reproducibility protocol

This document maps the manuscript validation claims to executable checks and archived artifacts.

## Reproducibility levels

### Level A — fast integrity verification

```bash
pytest -q
python verify_release.py
```

These commands verify the software tests, checkpoint integrity, frozen configuration, archived manuscript numbers, deterministic protocol/reference checks, and publication-figure presence.

### Level B — deterministic AMDI validation

```bash
python experiments/01_deterministic_backend_regression.py --config configs/publication.json
python experiments/02_protocol_alignment_audit.py --config configs/publication.json
python experiments/03_uniform_reference_check.py --config configs/publication.json
```

Acceptance criteria:

- reconstructed field relative error below `1e-12`;
- identical adaptive tree;
- coefficient discrepancy below `1e-12`;
- protocol final relative difference equal to zero within numerical precision;
- protocol tree distance zero;
- uniform-reference basis size `1024` at `32×32`;
- monotone reference energy;
- all reference safeguards accepted.

### Level C — figure regeneration

Quantitative figures:

```bash
python make_final_metric_figures.py \
  --summary results/publication_summary_FINAL/publication_summary_FINAL.json \
  --outdir IMG_TEST
```

Four-region evaluation:

```bash
python make_four_region_figures.py \
  --checkpoint checkpoints/selected/best_policy.pt \
  --device cpu \
  --outdir IMG_TEST
```

No training is performed by either command.

### Level D — numerical reevaluation from frozen checkpoints

```bash
python reproduce_final_results.py --device cpu --outdir reproduced_results
```

This is the strongest referee-facing verification short of retraining. It re-evaluates every archived policy required for the main holdout, Pareto family, resolution transfer, reward ablation, and training-seed robustness.

## Exact dataset protocol

Clean-image seeds:

```text
train       0–15
validation  100–103
test        1000–1007
```

Gaussian-noise standard deviation:

```text
sigma = 0.08
```

The noisy image is clipped pointwise to `[0,1]`.

Noise seeds used for final evaluation:

| Evaluation | Noise seed |
|---|---|
| nine-case holdout, random multiscale | `30000 + image_seed` |
| four-region holdout | `137` |
| Pareto subset | `40000 + image_seed` |
| resolution transfer | `50000 + image_seed + n` |
| reward ablation | `60000 + image_seed` |
| training-seed robustness | `70000 + image_seed` |

The Pareto, reward-ablation, and robustness calculations use the fixed four-image subset `{1000,1001,1002,1003}`.

## Selected policy and checkpoint selection

The main policy has training seed `20260811`. Checkpoint selection was based on the largest mean validation return under deterministic action selection. Test data were excluded from model selection.

The selected policy is:

```text
checkpoints/selected/best_policy.pt
```

Its SHA-256 is checked by `verify_release.py`.

## Pareto policies

The occupancy weights are exactly:

```text
0.03, 0.08, 0.15, 0.25, 0.30, 0.60
```

All use the stabilized 400-update PPO protocol and the same training seed `20260811`.

## Independent training seeds

Robustness uses:

```text
20260811, 20260817, 20260823
```

The exact checkpoints used for the final aggregate are under:

```text
checkpoints/robustness/
```

## Local feature construction

The actor input for each candidate leaf contains six components:

1. norm of the three prospective tensor-product Haar detail coefficients of the observed image;
2. propagated-field gradient magnitude over the leaf support enlarged by a one-pixel halo;
3. propagated-field variance over the same halo support;
4. normalized hierarchy level;
5. local active-leaf occupancy;
6. normalized decision time.

The feature scales `s_q`, `s_g`, and `s_v` are the 0.95 quantiles of absolute raw training-feature values estimated from the first three post-propagation deterministic-AMDI decision states. The fixed regularizer is `epsilon_feat = 1e-12`.

## Adaptive action resolution

Local actions are:

```text
coarsen / retain / refine
```

Coarsening is executed only when all four active leaf siblings vote to coarsen. Refinement is subsequently applied to surviving leaves. Retention is always admissible.

## Terminal and trajectory metrics

For each test case:

```text
E_ref = terminal reference discrepancy
C_rel = terminal active representation / full representation
```

Switching is averaged over the complete trajectory. RMSE and SSIM are evaluated on the terminal reconstruction against the clean image.

## VAMPyR/MRCPP

VAMPyR/MRCPP is used only as an independent multiresolution-localization cross-check. It is not the production AMDI propagator and its effective levels are not interpreted as numerically equivalent to Haar refinement levels.
