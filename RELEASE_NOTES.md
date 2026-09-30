# Release notes — v1.1.0

This release aligns the public reproducibility archive with the revised manuscript.

## Added

- complete nine-case, six-decision action/tree/trajectory/energy audit;
- deterministic candidate energies and tree penalties;
- decision-1-only temporal control;
- validation-tuned retain-tree, top-`K`, and fixed-threshold controls;
- paired case-level holdout and transfer differences;
- common-four-image PPO/control frontiers;
- final Figure 2 and Figure 3 generation scripts;
- an explicit regression test that executes the learned tree update;
- cross-platform manifest generation and GitHub Actions verification.

## Updated

- all six manuscript figures with larger typography and marks, no panel titles, and bold panel identifiers;
- manuscript-to-data mapping, reproducibility instructions, validation status, citation metadata, and release verification;
- plotting scripts for the final journal presentation.

## Removed

- superseded trajectory-only and PPO-only frontier figures;
- macOS metadata, caches, compiled Python files, and duplicate packaging directories.

No numerical algorithm, frozen publication configuration, or selected checkpoint was changed.
