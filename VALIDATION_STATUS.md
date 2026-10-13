# Validation Status

## Core source provenance

The reusable `src/iiot_msresstgat/` package and tests were copied directly from the original private project repository rather than reconstructed from the manuscript text.

## Curation checks

- Core Python source was checked locally with `python -m compileall`.
- A dependency-light test subset completed with `20 passed`.
- The full graph-model test suite was not run in the curation container because `torch_geometric` is not installed there.
- Raw SWaT data and machine-local absolute paths are excluded from the publication-facing repository.

Before public release, the full suite will be rerun in the frozen PyTorch/PyTorch-Geometric environment and exact environment metadata will be added under `environment/`.
