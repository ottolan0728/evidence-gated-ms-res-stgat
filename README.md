# Evidence-Gated MS-Res-STGAT for Industrial Time-Series Anomaly Detection

Private reproducibility repository for the IEEE Access manuscript:

**Evidence-Gated MS-Res-STGAT for Industrial Time-Series Anomaly Detection**

This repository is being curated during peer review and is intended for public release upon publication.

## Scope

The repository supports the manuscript's post-review reproducibility package, including:

- the historical Fold-0 Evidence-Gated Model Promotion (EGMP) evidence boundary;
- the parameter-matched Mean/Mean capacity control (R3);
- same-protocol Fold-0 baselines (R4);
- post-review cross-fold evaluation on outer folds 1–4 (R5);
- promotion-gate sensitivity, graph sensitivity, and component ablation (R6);
- frozen configuration files and derived result tables where redistribution is permitted.

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
├── .gitignore
├── environment/
│   └── README.md
├── configs/
│   ├── R3_CAPACITY_CONTROL_FREEZE.yaml
│   ├── R4_BASELINE_BENCHMARK_FREEZE.yaml
│   └── R5_CROSSFOLD_FREEZE.yaml
├── results/
│   ├── r3_capacity/
│   ├── r4_baselines/
│   ├── r5_crossfold/
│   └── r6_sensitivity_ablation/
└── src/
    └── README.md
```

## Data availability

The SWaT A1/A2 data are subject to the dataset provider's access and licensing conditions and are **not redistributed** in this repository. Users must obtain the dataset from the authorized provider and comply with its license.

## Code availability

During peer review, the repository remains private while the frozen experimental code and execution adapters are curated into a publication-ready layout. The public release will include the non-proprietary reproducibility code, configuration artifacts, and derived result tables needed to reproduce the manuscript's reported analyses, subject to the SWaT license.

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
