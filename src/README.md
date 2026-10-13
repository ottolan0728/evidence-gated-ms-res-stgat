# Source Code

`src/iiot_msresstgat/` contains the reusable core implementation recovered directly from the version-controlled project repository used for the manuscript workflow.

The package includes data handling, graph construction, MS-Res-STGAT model definitions, training/evaluation helpers, threshold-policy logic, reproducibility helpers, and supporting experiment utilities.

Post-review R3–R6 execution added additional role-isolation, fail-closed guards, capacity-matched controls, same-protocol baseline adapters, and cross-fold resource controls. Those execution-specific controllers are documented separately under `review_experiments/` while they are curated for public release.

Raw SWaT data are not included.
