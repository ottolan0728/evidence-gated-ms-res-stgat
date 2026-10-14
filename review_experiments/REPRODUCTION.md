# Path-Neutral Post-Review Reproduction

`reproduce_post_review.py` is the publication-facing entry point for R3–R6.

It is **not** represented as the byte-identical peer-review controller. The historical controllers were bound to machine-local paths, approval records, and SHA-256 execution bundles. The wrapper instead reuses the retained scientific modules with path-neutral arguments.

## Data contract

The wrapper accepts role-isolated **train** and **validation** NPZ shards and a metadata JSON. It refuses test/reserved roles.

Required provenance checks include:
- 51 aligned sensors;
- 60-s windows;
- stride 5 for training and 1 for validation;
- 59-s boundary purge;
- no cross-event windows;
- payload and partition-provenance hashes.

SWaT data are not included.

## Example

```bash
python review_experiments/reproduce_post_review.py \
  --stage r5 \
  --model dasp \
  --fold 1 \
  --seed 11 \
  --data-root /path/to/authorized_role_shards \
  --metadata /path/to/DATA_IDENTITY_METADATA.json \
  --cache-root outputs/cache \
  --output outputs/r5_dasp_outer1_seed11 \
  --device cuda
```

Frozen stages:
- `r3`: parameter-matched capacity control, Fold 0
- `r4`: same-protocol Fold-0 baselines
- `r5`: outer folds 1–4
- `r6c`: graph sensitivity, Fold 0
- `r6d`: component ablation, Fold 0

Use only model/fold/seed combinations listed in `configs/frozen/`.

A successful run is a **publication reproduction run** and does not retroactively alter the historical Fold-0 decision.
