import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    deterministic_algorithms: bool = False,
    deterministic_warn_only: bool = False,
) -> None:
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        deterministic_algorithms, warn_only=deterministic_warn_only
    )


def capture_rng_state(generator: torch.Generator | None = None) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator": generator.get_state() if generator is not None else None,
    }


def restore_rng_state(state: dict, generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if generator is not None and state.get("generator") is not None:
        generator.set_state(state["generator"].cpu())
