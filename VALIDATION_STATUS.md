# Validation status

Release status: **complete referee-ready numerical archive**.

## Automated checks

```text
pytest -q:               22 passed
python verify_release.py: PASS
```

The release verifier covers checkpoint hashes, frozen configuration values, original publication aggregates, the revised action/energy audit, validation-tuned controls, deterministic/reference records, all six figure pairs, citation metadata, and the SHA-256 manifest.

## Deterministic regression checks

Backend comparison:

```text
field relative difference:        0.0
tree identical:                   true
basis size direct/wrapper:        190 / 190
reference max coefficient error:  0.0
cross-platform maximum error:     2.1684e-19
```

Within-backend protocol comparison:

```text
final relative difference: 0.0
C_rel baseline/control:     0.15625 / 0.15625
tree distance:              0
pass:                       true
```

The two occupancy values reported for the separate manuscript regression checks are not interchangeable.

## Uniform reference

```text
number of states:          7
full basis size:           1024 / 1024
energy monotone:           true
all safeguards accepted:   true
pass:                      true
```

## Section 5 action and energy audit

```text
holdout cases:                         9
deterministic decisions:              54
deterministic accepted refinements:    0
deterministic coarsening groups:       4
learned refinements at decision 1:   383
learned refinements at decision 2:    10
learned refinements thereafter:        0
learned accepted refinements total:  393
learned energy-increasing adaptations: 15
```

The decision-1-only control reaches `E_ref = 0.13742`, compared with `0.13657` for unrestricted deployment.

## Validation-tuned controls

```text
Learned AMDI:               E_ref 0.13657, C_rel 0.26660
Top-K, matched occupancy:   E_ref 0.14175, C_rel 0.24414
Threshold, matched:         E_ref 0.13792, C_rel 0.26042
```

Control choices are frozen on validation images before holdout evaluation. The threshold control is close to PPO at comparable occupancy and slightly improves the mean image metrics.

## Figures

Exactly six manuscript figures are archived as valid PDF and 600-dpi PNG pairs under `IMG/`. They have enlarged typography and marks, no panel titles, and bold panel identifiers.
