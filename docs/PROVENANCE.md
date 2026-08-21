# Numerical provenance

The deterministic `amdi/` numerical core in this release was retained from the companion AMDI validation archive used to build the learned extension.

Source archive recorded during development:

```text
amdi_validation_v0.4.zip
SHA-256: f6ebe3d477f617f647a6cb523a9ad72196b753b1fe4a1e5ebed348315139b95a
```

The public referee release keeps the deterministic numerical core itself and the regression tests required to verify it. Superseded development archives, obsolete checkpoints, old ordering experiments, caches, and intermediate result folders have deliberately been removed.

The production AMDI validation backend is the adaptive tensor-product Haar implementation under `amdi/`. VAMPyR/MRCPP is used only for the independent localization cross-check.
