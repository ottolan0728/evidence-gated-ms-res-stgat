import random

import numpy as np
import torch
from torch_geometric.data import Data

from iiot_msresstgat.publication_experiment import build_publication_loaders
from iiot_msresstgat.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


def test_rng_state_round_trip():
    generator = torch.Generator().manual_seed(11)
    state = capture_rng_state(generator)
    expected = (random.random(), np.random.random(), torch.rand(1), torch.rand(1, generator=generator))
    restore_rng_state(state, generator)
    actual = (random.random(), np.random.random(), torch.rand(1), torch.rand(1, generator=generator))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    assert torch.equal(expected[3], actual[3])


def test_seed_everything_can_enable_strict_determinism():
    try:
        seed_everything(11, deterministic_algorithms=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


def test_validation_only_loader_does_not_require_test_dataset():
    sample = Data(
        x=torch.zeros(2, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor(0.0),
    )
    loaders = build_publication_loaders(
        {"train": [sample], "validation": [sample]},
        batch_size=1,
        generator=torch.Generator().manual_seed(11),
    )
    assert set(loaders) == {"train", "validation"}
