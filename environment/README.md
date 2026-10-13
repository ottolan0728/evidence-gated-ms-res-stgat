# Environment

The final public release will include an environment lock file generated from the frozen experiment environment.

To avoid inventing software-version claims, exact package pins are not being reconstructed from memory in this private review-stage repository.

Before public release:
1. recover the exact frozen environment metadata from the execution artifacts;
2. generate a pinned `requirements.txt` and/or `environment.yml`;
3. verify one clean installation and smoke test;
4. record hardware/runtime notes in this directory.
