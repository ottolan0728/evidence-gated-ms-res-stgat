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
- reusable core MS-Res-STGAT source code and lightweight tests;
- frozen/publication-facing configurations and derived result tables where redistribution is permitted.

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

## Data availability

The SWaT A1/A2 data are subject to the dataset provider's access and licensing conditions and are **not redistributed**. Users must obtain authorized access from the provider and comply with the applicable license.

## Code status during review

The reusable core project code and tests have been copied from the version-controlled project archive into this private repository.

Post-review R3–R6 execution used additional role-isolation, authorization, and audit controllers. Those exact controllers were tightly bound to machine-local paths and SHA-256 code bundles. They are being curated separately so that no machine-specific paths are published and no modified file is misrepresented as the exact hashed execution artifact. Frozen configs and manuscript-facing result summaries are already retained here.

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

## Contact

Corresponding author: Yi-Kai Su  
Repository owner / manuscript author: Chun-Jan Tseng (`ottolan0728`)
