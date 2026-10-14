# Evidence-Gated MS-Res-STGAT for Industrial Time-Series Anomaly Detection

Private reproducibility repository for the IEEE Access manuscript:

**Evidence-Gated MS-Res-STGAT for Industrial Time-Series Anomaly Detection**

This repository is being curated during peer review and is intended for public release upon publication.

## Scope

The repository supports:
- historical Fold-0 Evidence-Gated Model Promotion (EGMP) evidence;
- parameter-matched capacity control (R3);
- same-protocol Fold-0 baselines (R4);
- post-review cross-fold evaluation on outer folds 1–4 (R5);
- promotion-gate sensitivity, graph sensitivity, and component ablation (R6);
- reusable core MS-Res-STGAT source code and tests;
- a path-neutral publication reproduction wrapper for R3–R6;
- publication-facing configurations, environment documentation, and derived result tables where redistribution is permitted.

## Important evidence boundary

The historical Fold-0 decision remains:

`STOP-SUPERVISED-REDESIGN`

Reviewer-requested R3–R6 analyses are **post-review evidence**. They characterize capacity, baseline competitiveness, partition sensitivity, gate sensitivity, graph assumptions, and component dependence, but do not retroactively alter the historical Gate-4B decision.

The five deterministic seeds `{11, 22, 33, 44, 55}` are computational-stability repetitions, not independent population samples.

## Repository layout

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── DATA_POLICY.md
├── VALIDATION_STATUS.md
├── PUBLIC_RELEASE_CHECKLIST.md
├── pyproject.toml
├── LICENSE
├── configs/
├── results/
├── environment/
├── scripts/
├── src/iiot_msresstgat/
├── tests/
└── review_experiments/
```

## Reproduction entry point

A path-neutral publication reproduction wrapper is available at:

`review_experiments/reproduce_post_review.py`

It supports R3, R4, R5, R6C, and R6D using explicit command-line paths and role-isolated train/validation shards. Test/reserved roles are not accepted.

See:
- `review_experiments/REPRODUCTION.md`
- `REPRODUCIBILITY.md`
- `environment/ENVIRONMENT_STATUS.md`

The wrapper is intentionally labeled as a **publication reproduction wrapper**. It is not represented as the byte-identical historical peer-review controller, because the historical controllers were bound to machine-local paths, authorization artifacts, and SHA-256 code bundles.

## Data availability

The SWaT A1/A2 data are subject to the dataset provider's access and licensing conditions and are **not redistributed**. Users must obtain authorized access from the provider and comply with the applicable license.

## Environment status

An exact historical R3–R6 dependency lock is not fabricated. The repository provides a compatibility specification in `pyproject.toml` and a runtime-capture utility:

`environment/capture_environment.py`

Before public release, the authors will capture and verify a publication reproduction environment and run the full PyTorch/PyTorch-Geometric test suite.

## Manuscript-level headline results

Historical Fold-0 DASP:
- mean macro scenario PR-AUC: `0.5671`
- worst-seed macro scenario PR-AUC: `0.5226`
- mean constrained event recall: `0.4286`
- worst-seed constrained event recall: `0.2857`
- historical decision: `STOP-SUPERVISED-REDESIGN`

Post-review same-protocol Fold-0 benchmark:
- highest mean PR-AUC: TCN (`0.5784`)
- highest mean constrained event recall: DASP (`0.4286`)

These endpoints are intentionally reported separately.

## Review-stage status

The repository is suitable for **private peer-review use**. Before changing visibility to Public, complete the checks in `PUBLIC_RELEASE_CHECKLIST.md`.

## Contact

Corresponding author: Yi-Kai Su  
Repository owner / manuscript author: Chun-Jan Tseng (`ottolan0728`)
