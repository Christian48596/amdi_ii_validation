# GitHub / referee release checklist

- [x] Frozen configuration is `configs/publication.json`.
- [x] Selected, Pareto, robustness, and reward-ablation checkpoints are included.
- [x] Original publication aggregates are included.
- [x] Complete Section 5 action/energy audit is included.
- [x] Validation-selected controls and paired case-level results are included.
- [x] Transfer controls at `32x32`, `64x64`, and `128x128` are included.
- [x] Exactly six final manuscript figures are included as PDF and PNG.
- [x] Every final figure has a regeneration script.
- [x] Terminal RMSE/SSIM clipping and VAMPyR interpretation are documented.
- [x] `pytest -q` passes: 22 tests.
- [x] `python verify_release.py` passes.
- [x] `MANIFEST.sha256` verifies.
- [x] MIT `LICENSE` is included.
- [x] `CITATION.cff` matches the manuscript authors.
- [x] macOS metadata, caches, obsolete figures, and duplicate folders are absent.
- [ ] Add the final GitHub URL and archival DOI after they are assigned.
