# Learned Adaptive Multiresolution Diffusion Imaging

Reproducibility repository for the manuscript **“Learned Adaptive Multiresolution Diffusion Imaging.”**

This release is organized for independent referee verification. It contains the deterministic AMDI numerical core, the Learned AMDI policy and PPO implementation, the frozen publication configuration, all checkpoints used in the manuscript, the final numerical data tables, and scripts that regenerate every publication figure contained in this repository.

## What the repository verifies

Learned AMDI preserves the deterministic AMDI **propagate → adapt** outer-iteration ordering. The diffusion propagator is unchanged; learning controls the post-propagation adaptive multiresolution update.

The release includes three explicit deterministic checks:

1. **Backend regression:** the modular deterministic wrapper reproduces the retained deterministic AMDI implementation to machine precision, with identical adaptive trees.
2. **Protocol alignment:** the baseline and modular deterministic control produce zero final field difference, identical terminal occupancy, and zero tree distance.
3. **Uniform reference:** the complete Haar representation has 1024 active degrees of freedom at `32×32`, with monotone energy and accepted safeguards.

The manuscript comparisons are then reported only for:

- **AMDI** — deterministic adaptive baseline;
- **Learned AMDI** — learned post-propagation adaptive selector;
- **Uniform AMDI reference** — full Haar representation used as the algorithmic reference for `E_ref`.

The deterministic control is a verification pathway, not a separate method in the manuscript tables.

## Referee quick start

### 1. Create the environment

Recommended Conda installation:

```bash
conda env create -f environment.yml
conda activate learned-amdi
pip install -e .
```

On a headless Linux system:

```bash
export MPLBACKEND=Agg
```

If VAMPyR is not available for the local platform, the complete AMDI/Learned-AMDI core, tests, checkpoint verification, and all non-VAMPyR figures can be run with:

```bash
conda env create -f environment-core.yml
conda activate learned-amdi-core
pip install -e .
```

The archived VAMPyR/MRCPP localization data and figure remain included under `results/validation/` and `IMG/`.

### 2. Run the unit tests

```bash
pytest -q
```

The release contains **21 tests** covering the deterministic AMDI algebra, hierarchy operations, energy behavior, feature construction, action masking, unanimous sibling coarsening, reference construction, PPO joint probabilities, and Learned AMDI backend behavior.

### 3. Run the fast release verification

```bash
python verify_release.py
```

This checks:

- SHA-256 integrity of every publication checkpoint;
- the frozen publication configuration;
- the numerical values quoted in the manuscript;
- deterministic protocol/reference validation records;
- presence of all publication figures.

Expected output:

```text
Learned AMDI release verification: PASS
- checkpoint hashes: PASS
- frozen publication configuration: PASS
- manuscript numerical values: PASS
- deterministic protocol/reference checks: PASS
- publication figures present: PASS
```

### 4. Re-run the deterministic checks

```bash
python experiments/01_deterministic_backend_regression.py \
    --config configs/publication.json

python experiments/02_protocol_alignment_audit.py \
    --config configs/publication.json

python experiments/03_uniform_reference_check.py \
    --config configs/publication.json
```

Cross-platform floating-point arithmetic can produce a coefficient difference at the level of machine roundoff. For example, the Linux verification environment used to package this release produced a maximum coefficient difference of approximately `2.2e-19`; the archived manuscript run on the reference environment recorded `0.0`. Both are far below the regression tolerance of `1e-12`, and the reconstructed field and adaptive tree agree exactly in the regression check.

## Regenerate the manuscript figures

### Quantitative figures

No training or simulation is performed by this command. It reads the frozen publication summary only.

```bash
python make_final_metric_figures.py \
    --summary results/publication_summary_FINAL/publication_summary_FINAL.json \
    --outdir IMG_TEST
```

This generates:

```text
Fig_pareto_accuracy_complexity.pdf
Fig_resolution_transfer.pdf
Fig_robustness_ablation.pdf
Fig_vampyr_localization.pdf
```

### Four-region reconstruction and trajectory diagnostics

This is evaluation only; it loads the selected policy checkpoint and does not retrain the policy.

CPU:

```bash
python make_four_region_figures.py \
    --checkpoint checkpoints/selected/best_policy.pt \
    --device cpu \
    --outdir IMG_TEST
```

Apple Silicon with MPS:

```bash
python make_four_region_figures.py \
    --checkpoint checkpoints/selected/best_policy.pt \
    --device mps \
    --outdir IMG_TEST
```

This generates:

```text
Fig_four_region_reconstruction_and_refinement.pdf
Fig_four_region_trajectory_diagnostics.pdf
```

The checked publication figures are retained under [`IMG/`](IMG/).

## Reproduce the numerical summaries from the archived checkpoints

To rerun the fixed holdout, Pareto, resolution-transfer, reward-ablation, and three-seed robustness evaluations **without training**:

```bash
python reproduce_final_results.py \
    --device cpu \
    --outdir reproduced_results
```

On Apple Silicon, `--device mps` can substantially reduce runtime. A full CPU reevaluation includes the 64×64 and 128×128 transfer cases and can take several minutes. The script prints progress by validation block, uses exactly the image seeds, noise seeds, policy checkpoints, and evaluation subsets used for the publication dataset, and then compares the recomputed values with the archived manuscript values. It reports `PASS` when they agree within floating-point tolerance.

The VAMPyR/MRCPP localization analysis is intentionally separate because it requires the optional VAMPyR dependency:

```bash
python experiments/10_vampyr_localization_crosscheck.py \
    --checkpoint checkpoints/selected/best_policy.pt \
    --device cpu
```

VAMPyR/MRCPP is an **independent localization cross-check**, not the production AMDI propagator.

## Frozen publication protocol

The definitive configuration is:

```text
configs/publication.json
```

Principal values:

| Quantity | Value |
|---|---:|
| training resolution | `32×32` |
| AMDI outer iterations | `6` |
| selected training seed | `20260811` |
| actor hidden layers | `[64, 64]` |
| critic hidden layers | `[64, 64]` |
| activation | `tanh` |
| `gamma` | `1.0` |
| GAE `lambda` | `0.95` |
| PPO clip | `0.10` |
| critic coefficient | `0.5` |
| entropy coefficient | `0.02` |
| Adam learning rate | `5e-5` |
| PPO epochs/update | `4` |
| trajectories/update | `16` |
| PPO updates | `400` |
| max gradient norm | `0.5` |
| `lambda_err` | `1.0` |
| selected `lambda_occ` | `0.15` |
| `lambda_sw` | `0.02` |

AMDI parameters are also frozen in the same JSON file.

## Publication numerical results

The definitive machine-readable dataset is:

```text
results/publication_summary_FINAL/publication_summary_FINAL.json
```

The nine-case holdout values are:

| Method | `E_ref` | `C_rel` | RMSE | SSIM | mean switching |
|---|---:|---:|---:|---:|---:|
| AMDI | 0.17496 | 0.13737 | 0.05391 | 0.80851 | 0.000217 |
| Learned AMDI | 0.13657 | 0.26660 | 0.05306 | 0.79899 | 0.02132 |
| Uniform AMDI reference | 0 | 1 | 0.07141 | 0.70150 | 0 |

The selected Learned AMDI policy therefore decreases mean terminal reference error by approximately **21.9%** relative to AMDI, while using a larger terminal representation. The repository does **not** interpret occupancy as wall-clock computational cost.

For the fixed four-image Pareto subset, the lowest RMSE and highest SSIM occur at `lambda_occ = 0.25`.

## Data splits and deterministic seeds

Image seeds:

```text
training:   0 ... 15
validation: 100 ... 103
test:       1000 ... 1007
```

The publication evaluations additionally use fixed deterministic noise-seed conventions documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Repository structure

```text
.
├── amdi/                         deterministic AMDI numerical core
├── learned_amdi/                 learned selector, features, PPO, evaluation
├── checkpoints/
│   ├── selected/                 selected policy used for main comparisons
│   ├── pareto/                   six occupancy-penalty policies
│   ├── robustness/               three independent training seeds
│   └── ablation/                 reward-ablation policies
├── configs/
│   └── publication.json          frozen publication configuration
├── experiments/                  validation/training/evaluation entry points
├── results/
│   ├── publication_summary_FINAL/ definitive manuscript dataset
│   └── validation/               deterministic/reference validation records
├── IMG/                          checked publication figures (PDF + PNG)
├── tests/                        unit and regression tests
├── make_final_metric_figures.py
├── make_four_region_figures.py
├── reproduce_final_results.py
├── verify_release.py
└── REPRODUCIBILITY.md
```

## Important interpretation limits

This repository supports the claims made in the manuscript, specifically:

- exact protocol alignment of the deterministic regression pathways;
- an accuracy–representation family controlled by `lambda_occ`;
- lower selected-policy `E_ref` and slightly lower RMSE on the nine-case holdout, with larger occupancy and slightly lower SSIM;
- resolution transfer to `64×64` and `128×128` without retraining;
- robustness across three independent training seeds;
- the essential role of the occupancy reward term;
- a coupled, nonmonotonic effect of the switching term;
- qualitative VAMPyR/MRCPP corroboration of spatial localization.

The repository does **not** claim:

- wall-clock or memory speedup from `C_rel` alone;
- uniform dominance of Learned AMDI over deterministic AMDI;
- quantitative equivalence between Haar refinement levels and VAMPyR/MRCPP effective levels;
- transfer across diffusion parameters not tested here.

## Provenance

The retained deterministic `amdi/` core originated from the companion AMDI validation archive used during development. Its source-archive SHA-256 is recorded in [`docs/PROVENANCE.md`](docs/PROVENANCE.md). The public release keeps the numerical core and the exact regression tests required for verification, while obsolete intermediate checkpoints and superseded result folders have been removed.

## License

A software license has deliberately **not** been selected automatically. Before making the repository public, the authors should choose the desired license. A private repository shared with referees can be used as-is.
