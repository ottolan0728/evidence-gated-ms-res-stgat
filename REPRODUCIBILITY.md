# Reproducibility Notes

## Evidence classes

The repository distinguishes four evidence classes:

1. **Historical Fold-0 evidence** — evidence used for the original Gate-4B decision.
2. **Post-review capacity/baseline evidence** — R3 and R4.
3. **Post-review cross-fold evidence** — R5.
4. **Post-review sensitivity/ablation evidence** — R6.

Post-review evidence must not be interpreted as a retroactive promotion stage.

## Common endpoint

The manuscript evaluates two distinct endpoint families:

- threshold-free ranking using macro scenario PR-AUC;
- alarm-constrained event recovery under the frozen alarm policy.

Alarm feasibility requires:
- false-alarm episodes/hour <= 1;
- normal alarm duty <= 1%.

For every model–seed evaluation, the common endpoint evaluates 501 score/quantile threshold candidates plus a no-alarm candidate. Among feasible candidates, the selector maximizes:
1. event recall;
2. then window-level F1;
3. then minimizes normal alarm duty;
4. then prefers the larger threshold.

If no candidate is feasible, the constraints are not relaxed.

## Historical promotion criteria

- mean macro scenario PR-AUC gain >= 0.03;
- worst-seed PR-AUC gain >= 0.05;
- mean constrained event recall >= 0.60;
- worst-seed constrained event recall >= 0.50.

The original criteria and `STOP-SUPERVISED-REDESIGN` decision are immutable historical evidence.

## Seeds and partition interpretation

Seeds: `11, 22, 33, 44, 55`

Seeds quantify computational stability within a fixed partition. They are not independent statistical replicates.

In R5, the outer fold is the partition-replication unit. The rotating outer folds reuse the same SWaT process/attack corpus and are therefore not independent external datasets.

## Reproduction wrapper

The portable entry point is:

`review_experiments/reproduce_post_review.py`

The wrapper:
- accepts only role-isolated train/validation NPZ shards;
- validates the frozen 51-sensor, 60-s window, stride, and purge contract;
- checks shard hashes when provided;
- reconstructs training-only normalization and graph construction;
- supports R3, R4, R5, R6C, and R6D model/stage definitions;
- calls the retained core event metrics and alarm-constrained threshold selector;
- writes a structured `reproduction_result.json` including runtime metadata.

It is deliberately labeled as a **publication reproduction wrapper**, not as the byte-identical historical peer-review controller.

See `review_experiments/REPRODUCTION.md` for usage.

## Environment

The repository does not fabricate an exact historical dependency lock. The current evidence supports a compatibility specification and Python-3.12-era historical artifacts, but not a complete trustworthy historical `pip freeze` for all executed R3–R6 runs.

Use:

`environment/capture_environment.py`

to capture an exact publication-reproduction runtime before public release. See `environment/ENVIRONMENT_STATUS.md`.

## Data boundary

The SWaT A1/A2 raw data, role-isolated processed shards, raw windows, and checkpoints are not redistributed. Users must obtain authorized SWaT access separately.
