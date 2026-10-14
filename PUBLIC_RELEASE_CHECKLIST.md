# Public Release Checklist

Keep this repository **Private** during peer review unless the authors intentionally decide otherwise.

Before changing repository visibility to Public:

- [ ] Full test suite passes in a compatible PyTorch/PyTorch-Geometric environment.
- [ ] `environment/capture_environment.py` has generated `environment/runtime_inventory_publication.json`.
- [ ] A publication reproduction dependency lock has been generated and clearly labeled as such.
- [ ] At least one authorized-data path-neutral reproduction run completes successfully.
- [ ] Reproduction output agrees with the expected endpoint implementation and role boundaries.
- [ ] No SWaT raw data or processed licensed role shards are committed.
- [ ] No model checkpoint is committed unless redistribution is explicitly permitted.
- [ ] No local absolute paths, credentials, API tokens, emails containing secrets, or private authorization artifacts are committed.
- [ ] README, DATA_POLICY, REPRODUCIBILITY, CITATION.cff, and LICENSE are final.
- [ ] Repository URL in the accepted manuscript matches this repository.
- [ ] Release tag/version is created only after the final manuscript version is frozen.

## Important provenance rule

Do not describe a future publication reproduction dependency lock as the byte-identical historical R3–R6 environment unless an independently verifiable historical lock is recovered.

Do not describe the path-neutral wrapper as the exact historical authorization controller. It is a curated publication reproduction entry point.
