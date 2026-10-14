# Path-Neutral Post-Review Reproduction

`reproduce_post_review.py` is the publication-facing entry point for R3–R6.

It is **not** represented as the byte-identical peer-review controller. The historical controllers were bound to machine-local paths, approval records, and SHA-256 execution bundles. The wrapper instead preserves the reviewed scientific contracts while replacing machine-specific controller dependencies with explicit command-line inputs.

## Data contract

The wrapper accepts role-isolated **train** and **validation** NPZ shards plus a metadata JSON. It does not accept test/reserved roles.

The metadata and shards are checked against the retained protocol boundary, including:
- 51 aligned sensors;
- 60-s windows;
- stride 5 for training and 1 for validation;
- 59-s boundary purge;
- no cross-event windows;
- role-isolated storage;
- payload SHA-256 verification when supplied.

SWaT data are not included in this repository.

## Example

```bash
python review_experiments/reproduce_post_review.py \
  --stage r5 \
  --model dasp \
  --fold 1 \
  --seed 11 \
  --data-root /path/to/authorized_role_shards \
  --metadata /path/to/DATA_IDENTITY_METADATA.json \
  --output outputs/r5_dasp_outer1_seed11 \
  --device cuda
```

## Supported frozen stages

- `r3`: parameter-matched capacity control, Fold 0
- `r4`: same-protocol Fold-0 baseline benchmark
- `r5`: outer folds 1–4
- `r6c`: graph-construction sensitivity, Fold 0
- `r6d`: major-component ablation, Fold 0

The publication wrapper embeds the reviewed stage/model specifications needed for those runs. Compact publication-facing stage summaries remain under `configs/`.

## Output

Each run writes:

`reproduction_result.json`

including:
- stage/model/fold/seed identity;
- parameter count when available;
- best epoch and completed epochs for neural models;
- role map;
- graph/scaler/metadata hashes;
- ranking and alarm-constrained event endpoint values;
- runtime information.

## Interpretation

A successful rerun is a **publication reproduction run**. It does not retroactively modify the historical Fold-0 Gate-4B decision, and it does not turn the rotating SWaT outer folds into independent external samples.

The wrapper is intentionally not described as the exact historical authorization controller.
