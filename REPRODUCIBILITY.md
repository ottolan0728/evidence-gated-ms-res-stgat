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

For every model–seed evaluation, the common endpoint evaluates 501 threshold candidates under the frozen procedure plus a no-alarm candidate. Among feasible candidates, the selector maximizes:
1. event recall;
2. then window-level F1;
3. then minimizes normal alarm duty;
4. then prefers the larger threshold.

If no candidate is feasible, the run returns `NO_FEASIBLE_THRESHOLD`; the constraints are not relaxed.

## Historical promotion criteria

- mean macro scenario PR-AUC gain >= 0.03;
- worst-seed PR-AUC gain >= 0.05;
- mean constrained event recall >= 0.60;
- worst-seed constrained event recall >= 0.50.

The original criteria and STOP decision are immutable historical evidence.

## Seeds and partition interpretation

Seeds: `11, 22, 33, 44, 55`

Seeds quantify computational stability within a fixed partition. They are not independent statistical replicates.

In R5, the outer fold is the partition-replication unit. The rotating outer folds reuse the same SWaT process/attack corpus and are therefore not independent external datasets.
