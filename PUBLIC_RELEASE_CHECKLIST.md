# Public Release Checklist

The associated IEEE Access manuscript was accepted on 31 October 2026.

## Completed publication-facing checks

- [x] Repository URL in the accepted manuscript matches this repository.
- [x] README, DATA_POLICY, REPRODUCIBILITY, CITATION.cff, and LICENSE are present.
- [x] Core source provenance is documented.
- [x] Core Python source compiles.
- [x] Dependency-light test subset passed (20 tests).
- [x] Path-neutral reproduction wrapper compiles and exposes a working CLI.
- [x] Accepted-manuscript result tables are included.
- [x] Historical and post-review evidence are explicitly separated.
- [x] No SWaT raw data or processed licensed role shards are intentionally committed.
- [x] No model checkpoints are intentionally committed.
- [x] No publication-facing local absolute paths, credentials, API tokens, or private authorization artifacts were found in the curation scan.
- [x] Threshold results are explicitly described as optimized validation operating-point evidence rather than untouched operational test estimates.

## Reproduction items that remain environment-dependent

- [ ] Full PyTorch/PyTorch-Geometric test suite on an author-controlled compatible environment.
- [ ] Authorized-data end-to-end smoke reproduction through the path-neutral wrapper.
- [ ] Optional capture of a publication reproduction environment lock.

These items are **not** represented as historical evidence and are not required to interpret the accepted manuscript. They may be added later as repository maintenance improvements.

## Important provenance rule

Do not describe a future publication reproduction dependency lock as the byte-identical historical R3–R6 environment unless an independently verifiable historical lock is recovered.

Do not describe the path-neutral wrapper as the exact historical authorization controller. It is a curated publication reproduction entry point.

## Release status

**Approved for public release with documented reproducibility limitations.**
