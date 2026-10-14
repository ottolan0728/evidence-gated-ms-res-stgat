# Post-Review Experiment Code

R3–R6 were executed after peer review using audit-oriented controllers with role guards, frozen configurations, deterministic seeds, and SHA-256 bindings.

## Publication reproduction entry point

A path-neutral publication reproduction wrapper is now available:

`review_experiments/reproduce_post_review.py`

The wrapper supports the reviewer-requested R3, R4, R5, R6C, and R6D stages using explicit command-line paths and role-isolated train/validation shards. Test/reserved roles are not accepted.

Usage and the data contract are documented in:

`review_experiments/REPRODUCTION.md`

## Historical-controller boundary

The exact peer-review execution controllers were tightly coupled to machine-local paths, authorization artifacts, and original code-hash bundles. They are therefore **not copied verbatim** into this publication-facing repository.

Publishing those controllers unchanged would expose local paths; editing them would invalidate the hashes that identified the executed code. The path-neutral wrapper is therefore explicitly labeled as a **publication reproduction wrapper**, not the byte-identical historical controller.

The repository also provides:

- reusable core scientific code under `src/iiot_msresstgat/`;
- publication-facing stage specifications under `configs/`;
- manuscript-facing aggregate result tables under `results/`;
- environment-capture utilities under `environment/`;
- documentation of the historical/post-review evidence boundary.

No SWaT raw data, role shards, or model checkpoints are redistributed.
