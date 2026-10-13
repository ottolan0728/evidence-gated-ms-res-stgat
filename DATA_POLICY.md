# Data Policy

## SWaT

This repository does **not** redistribute the Secure Water Treatment (SWaT) A1/A2 raw dataset.

The data are subject to the access and licensing conditions of the dataset provider. Users wishing to reproduce the analyses must obtain authorized access directly from the provider and comply with the applicable license.

## Files that must never be committed

- SWaT raw data
- raw `.npy` / `.npz` windows derived directly from licensed data
- model checkpoints containing licensed-data-derived artifacts unless redistribution is explicitly permitted
- passwords, API keys, tokens, or credentials
- machine-specific absolute paths
- temporary caches and local experiment scratch files

Derived aggregate result tables that do not expose raw licensed observations may be included where appropriate.
