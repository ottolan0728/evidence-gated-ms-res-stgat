# Post-Review Experiment Code

R3–R6 were executed after peer review using audit-oriented controllers with role guards, frozen configurations, deterministic seeds, and SHA-256 bindings.

The exact execution controllers were tightly coupled to machine-local paths, authorization artifacts, and original code-hash bundles. They are therefore **not copied verbatim** into this publication-facing repository at this stage.

Publishing those controllers unchanged would expose local paths; editing them would invalidate the hashes that identified the executed code. The publication package therefore currently provides:

- the reusable core scientific code under `src/iiot_msresstgat/`;
- frozen/publication-facing configurations under `configs/`;
- manuscript-facing aggregate result tables under `results/`;
- documentation of the historical/post-review evidence boundary.

Before public release, a path-neutral reproduction wrapper will be verified against the frozen result summaries. It will be clearly labeled as a publication reproduction wrapper rather than the byte-identical historical controller.
