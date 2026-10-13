# Contributing

Thank you for considering a contribution. Please open an issue before making a
substantial change so that its scope, evidence boundary, and validation plan can
be agreed in advance.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

Contributions must not include SWaT data, derived row-level telemetry, model
checkpoints, predictions, credentials, or machine-specific absolute paths.

## Scientific changes

Changes that could affect reported results require a prospectively documented
hypothesis, frozen configuration, data boundary, seed policy, endpoint, and
failure rule. The released Gate 4B result is immutable historical evidence. New
model families or post-Gate-4B experiments must be labelled as a distinct study
phase and must not silently overwrite the released evidence snapshot.
