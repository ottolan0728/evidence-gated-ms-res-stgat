# Validation Status

## Publication-facing status

The associated IEEE Access manuscript was accepted for publication on 31 October 2026.

This repository is the publication-facing reproducibility companion. Its purpose is to expose the reusable scientific code, frozen configuration summaries, result tables, and a path-neutral reproduction entry point without redistributing licensed SWaT data or falsely reconstructing unavailable historical environment details.

## Core source provenance

The reusable `src/iiot_msresstgat/` package and tests were copied directly from the original private project archive rather than reconstructed from manuscript prose.

## Completed curation checks

- Core Python source checked with `python -m compileall`.
- Dependency-light test subset: **20 passed**.
- Path-neutral reproduction wrapper:
  - Python compilation: **PASS**
  - command-line `--help` smoke test: **PASS**
- Repository curation search found no publication-facing local `C:\\Users` path strings.
- No raw SWaT data are included.
- No model checkpoints are included.
- No local credentials, API tokens, or authorization artifacts are intentionally included.
- The wrapper's alarm-policy call was checked against the retained core function signature:
  `select_false_alarm_constrained_threshold(..., grid_size=501)`.
- The wrapper's graph call was checked against:
  `build_graph(train_values, feature_names, correlation_threshold, adjacent_stages_only)`.
- Accepted-manuscript result tables are represented in `results/`.
- Historical and post-review evidence classes remain explicitly separated.

## Reproduction limitations

A full end-to-end reproduction requires:
- authorized SWaT access;
- compatible PyTorch / PyTorch-Geometric installation;
- role-isolated train/validation shards satisfying the documented provenance contract.

The byte-identical historical R3–R6 package lock was not recovered and is not fabricated. A future reproduction runtime may be recorded using `environment/capture_environment.py`.

The validation threshold is selected on the same validation events used to report constrained event recall and alarm burden. Those quantities are therefore optimized validation operating-point evidence, not untouched operational test estimates.

## Public-release assessment

**PASS for publication-facing public release with the documented limitations above.**
