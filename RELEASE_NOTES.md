# Release Notes

## v1.0.0 — Accepted-manuscript reproducibility release

**Date:** 31 October 2026

The associated manuscript, **“Evidence-Gated MS-Res-STGAT for Industrial Time-Series Anomaly Detection,”** has been accepted for publication in IEEE Access.

This repository release contains publication-facing source code, configuration summaries, derived result tables, documentation, and a path-neutral reproduction wrapper.

### Scientific boundary

- The historical Fold-0 decision remains `STOP-SUPERVISED-REDESIGN`.
- R3–R6 are post-review analyses and do not retroactively modify the historical decision.
- Deterministic seeds are computational repetitions, not independent samples.
- Cross-fold partitions are dependent rotations of the same SWaT corpus, not external datasets.
- Alarm-constrained event-recall results are optimized validation operating-point evidence, not untouched operational test estimates.

### Data boundary

SWaT A1/A2 raw data, processed role shards, raw windows, and model checkpoints are not redistributed.

### Environment boundary

A byte-identical historical R3–R6 dependency lock was not recovered and is not fabricated. The repository provides compatibility metadata and a runtime-capture utility for publication reproduction environments.

### Citation

Final DOI, volume, issue, and page metadata will be added after assignment by IEEE.
