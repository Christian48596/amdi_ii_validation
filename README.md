# Learned Adaptive Multiresolution Diffusion Imaging

Reproducibility repository for the manuscript **“Learned Adaptive Multiresolution Diffusion Imaging.”**

Release `v1.1.0` contains the deterministic AMDI numerical core, the Learned AMDI implementation, the frozen publication configuration, every checkpoint used in the manuscript, case-level numerical results, validation-tuned controls, and scripts for the six publication figures.

## Scientific scope

Both methods use the same **propagate → adapt** ordering and the same fixed-tree AMDI coefficient update. Learning changes only the post-propagation selection of the admissible multiresolution tree.

The revised experiments establish the following.

- The deterministic wrapper reproduces the retained AMDI implementation to machine precision with identical trees.
- On the nine holdout cases, deterministic AMDI accepts no refinements in 54 decisions and behaves as a nearly static initial-tree method.
- Learned AMDI executes 383 refinements at decision 1 and 10 at decision 2, for 393 accepted refinements in total.
- Learned AMDI lowers mean terminal reference discrepancy from `0.17496` to `0.13657`, while terminal occupancy increases from `0.13737` to `0.26660`.
- The validation-tuned observed-detail threshold reaches `E_ref = 0.13792` at occupancy `0.26042` and gives slightly better RMSE and SSIM. The evidence therefore supports PPO as a learned allocation mechanism, not as a universally superior spatial selector.
- The decision-1-only control reaches `E_ref = 0.13742`; later policy decisions contribute little on this static benchmark.
- Learned tree changes can increase the AMDI energy. The fixed-tree safeguard does not imply monotone decay of the complete learned outer iteration.

Occupancy counts active degrees of freedom. It is not interpreted as wall-clock time or memory consumption.

## Referee quick start

```bash
conda env create -f environment-core.yml
conda activate learned-amdi-core
python -m pip install -e .
pytest -q
python verify_release.py
```

Expected status:

```text
22 passed
Learned AMDI release verification: PASS
```

`environment.yml` additionally requests VAMPyR/MRCPP. The core calculations, checkpoint tests, archived VAMPyR results, and all manuscript figures remain available without installing VAMPyR.

## Frozen publication configuration

The definitive configuration is [`configs/publication.json`](configs/publication.json). The selected policy is [`checkpoints/selected/best_policy.pt`](checkpoints/selected/best_policy.pt).

Principal settings are:

| Quantity | Value |
|---|---:|
| training resolution | `32×32` |
| AMDI outer iterations | `6` |
| selected training seed | `20260811` |
| actor/critic hidden layers | `[64, 64]` |
| PPO updates | `400` |
| trajectories per update | `16` |
| learning rate | `5e-5` |
| selected `lambda_occ` | `0.15` |
| `lambda_sw` | `0.02` |

## Archived numerical results

| Method | `E_ref` | `C_rel` | RMSE | SSIM |
|---|---:|---:|---:|---:|
| Deterministic AMDI | 0.17496 | 0.13737 | 0.05391 | 0.80851 |
| Learned AMDI | 0.13657 | 0.26660 | 0.05306 | 0.79899 |
| Retain initial tree | 0.17494 | 0.13867 | 0.05392 | 0.80813 |
| Top-`K`, validation-occupancy matched | 0.14175 | 0.24414 | 0.05290 | 0.80628 |
| Threshold, validation-occupancy matched | 0.13792 | 0.26042 | 0.05266 | 0.80195 |

The complete records are stored in:

```text
results/publication_summary_FINAL/
results/11_section5_diagnostics/
results/12_section5_controls/
results/validation/
```

The control parameters were selected on validation images `100–103` and frozen before holdout testing. Case-level paired differences are retained rather than only aggregate means.

## Re-run the Section 5 audit and controls

These commands perform evaluation only and use the selected checkpoint:

```bash
python experiments/11_action_energy_audit.py --device cpu --include-transfer
python experiments/12_selector_controls.py --device cpu --include-transfer
```

The first command records proposed and executed actions separately, active degrees of freedom before and after adaptation, all deterministic candidate scores, trajectory errors, occupancy, switching, and energies. The second performs validation-only control selection and evaluates retain-tree, top-`K`, and fixed-threshold controls on the frozen holdout, Pareto, and transfer cases.

## Regenerate the six manuscript figures

```bash
python make_all_manuscript_figures.py --device cpu --outdir IMG_TEST
```

The command writes the six checked manuscript figures as vector PDF and
600-dpi PNG pairs.  By default, the four-region layout uses the archived
selected-checkpoint panels under `figure_assets/`.  To recompute those panels
from the selected checkpoint before plotting, add
`--recompute-four-region`.  The checked outputs are under [`IMG/`](IMG/).

## Other reproducibility levels

Deterministic regression:

```bash
python experiments/01_deterministic_backend_regression.py --config configs/publication.json
python experiments/02_protocol_alignment_audit.py --config configs/publication.json
python experiments/03_uniform_reference_check.py --config configs/publication.json
```

Checkpoint-only reevaluation of the original holdout, PPO Pareto family, transfer, reward ablation, and policy-seed robustness:

```bash
python reproduce_final_results.py --device cpu --outdir reproduced_results
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) for the exact data and manuscript mapping.

## Interpretation limits

This archive does not establish a runtime or memory speedup from occupancy, superior spatial selection relative to the tuned threshold rule, or a substantive sequential advantage on the static benchmark. Transfer controls are not occupancy matched at the two finer resolutions. VAMPyR/MRCPP is an order-five projection of the clean analytic target at precision `10^-3` and maximum depth eight; it is a qualitative localization diagnostic, not validation of the learned trajectory.

## Repository structure

```text
.
├── amdi/                          deterministic AMDI core
├── learned_amdi/                  learned selector and PPO implementation
├── checkpoints/                   selected, Pareto, robustness, and ablation policies
├── configs/publication.json       frozen publication configuration
├── experiments/                   executable validation and evaluation studies
├── results/
│   ├── publication_summary_FINAL/ original publication aggregates
│   ├── 11_section5_diagnostics/   action, tree, trajectory, and energy audit
│   ├── 12_section5_controls/      tuned controls, paired tests, and frontiers
│   └── validation/                deterministic and VAMPyR records
├── IMG/                            six final figure pairs
├── tests/                          unit and regression tests
├── docs/                           protocol, provenance, and results map
├── verify_release.py               fast numerical and file-integrity check
└── tools/generate_manifest.py      cross-platform SHA-256 manifest generator
```

## License and citation

The software is released under the MIT License; see [`LICENSE`](LICENSE). Citation metadata are provided in [`CITATION.cff`](CITATION.cff).
