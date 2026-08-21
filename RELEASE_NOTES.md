# Referee release notes — v1.0.0

This package is a cleaned reproducibility release prepared from the development archive.

Removed before release:

- superseded checkpoints and intermediate training experiments;
- obsolete adapt-before-propagate development artifacts;
- duplicate result folders;
- caches, compiled Python files, macOS metadata, and package-build metadata;
- internal manuscript-development naming from the public-facing code and data.

Retained:

- deterministic AMDI numerical core;
- Learned AMDI implementation;
- all final publication checkpoints;
- final numerical summaries and validation records;
- unit/regression tests;
- publication figure scripts and checked figures;
- optional VAMPyR/MRCPP localization cross-check.

The release changes repository organization and public naming only; it does not alter the numerical algorithms or publication checkpoints.
