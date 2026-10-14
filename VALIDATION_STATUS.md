# Validation Status

## Core source provenance

The reusable `src/iiot_msresstgat/` package and tests were copied directly from the original private project archive rather than reconstructed from manuscript prose.

## Completed curation checks

- Core Python source checked with `python -m compileall`.
- Dependency-light test subset: **20 passed**.
- Path-neutral reproduction wrapper:
  - Python compilation: **PASS**
  - command-line `--help` smoke test: **PASS**
- Repository curation search found no publication-facing `C:\\Users` path strings.
- Raw SWaT data are excluded.
- The wrapper's alarm-policy call was checked against the retained core function signature:
  `select_false_alarm_constrained_threshold(..., grid_size=501)`.
- The wrapper's graph call was checked against:
  `build_graph(train_values, feature_names, correlation_threshold, adjacent_stages_only)`.

## Still required before public release

- Run the full PyTorch/PyTorch-Geometric test suite in a compatible environment.
- Capture the exact **publication reproduction environment** under `environment/`.
- Run at least one authorized-data smoke reproduction through the path-neutral wrapper.
- Verify that no licensed SWaT payload, raw window, checkpoint, local credential, or machine-specific path is included.
- Review the final repository snapshot before changing visibility from Private to Public.

The repository is suitable for **private peer-review use** in its present state.
