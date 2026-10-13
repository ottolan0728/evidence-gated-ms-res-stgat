import torch
from torch_geometric.data import Batch, Data

import pytest
from torch.utils.data import Subset

from iiot_msresstgat.models import (
    MSResSTGAT,
    USADModel,
    build_model,
    temporal_statistics,
)
from iiot_msresstgat.usad_experiment import normal_only_subset
from iiot_msresstgat.publication_experiment import limited_balanced_subset


def test_model_forward_and_backward():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = Batch.from_data_list(
        [
            Data(x=torch.randn(3, 12), edge_index=edge_index, y=torch.tensor(0.0)),
            Data(x=torch.randn(3, 12), edge_index=edge_index, y=torch.tensor(1.0)),
        ]
    )
    model = MSResSTGAT(hidden_dim=16, heads=4, kernels=(3, 5), dropout=0.0)
    output = model(batch)
    assert output["logits"].shape == (2,)
    output["logits"].sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_explicit_mean_pooling_is_exactly_control_equivalent():
    torch.manual_seed(9)
    control = MSResSTGAT(hidden_dim=16, heads=4, kernels=(3, 5), dropout=0.0)
    torch.manual_seed(9)
    explicit = MSResSTGAT(
        hidden_dim=16,
        heads=4,
        kernels=(3, 5),
        dropout=0.0,
        temporal_pooling="mean",
        sensor_pooling="mean",
    )
    assert control.state_dict().keys() == explicit.state_dict().keys()
    for name, value in control.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = Batch.from_data_list(
        [Data(x=torch.randn(3, 12), edge_index=edge_index, y=torch.tensor(0.0))]
    )
    assert torch.equal(control(batch)["logits"], explicit(batch)["logits"])


@pytest.mark.parametrize(
    ("temporal_pooling", "sensor_pooling"),
    [
        ("mean_max_std", "mean"),
        ("mean", "mean_max_std"),
        ("mean_max_std", "mean_max_std"),
    ],
)
def test_statistics_pooling_variants_have_finite_gradients(
    temporal_pooling, sensor_pooling
):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = Batch.from_data_list(
        [
            Data(x=torch.ones(3, 12), edge_index=edge_index, y=torch.tensor(0.0)),
            Data(x=torch.ones(3, 12), edge_index=edge_index, y=torch.tensor(1.0)),
        ]
    )
    model = MSResSTGAT(
        hidden_dim=16,
        heads=4,
        kernels=(3, 5),
        dropout=0.0,
        temporal_pooling=temporal_pooling,
        sensor_pooling=sensor_pooling,
    )
    output = model(batch)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    output["logits"].sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_temporal_statistics_are_finite_for_zero_variance_input():
    pooled = temporal_statistics(torch.ones(2, 3, 5))
    assert pooled.shape == (2, 9)
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled[:, 6:], torch.full((2, 3), 1e-4))


@pytest.mark.parametrize("name", ["tcn", "gru", "lstm", "gcn", "gatv2", "gcn_gru"])
def test_neural_baseline_forward_and_backward(name):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = Batch.from_data_list(
        [
            Data(x=torch.randn(3, 12), edge_index=edge_index, y=torch.tensor(0.0)),
            Data(x=torch.randn(3, 12), edge_index=edge_index, y=torch.tensor(1.0)),
        ]
    )
    model = build_model(
        {"name": name, "hidden_dim": 16, "heads": 4, "dropout": 0.0}, num_nodes=3
    )
    output = model(batch)
    assert output["logits"].shape == (2,)
    output["logits"].sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("use_residual", "spatial_encoder", "kernels"),
    [(False, "gatv2", (3, 5)), (True, "mean_pool", (3,)), (True, "gatv2", (3,))],
)
def test_proposed_model_ablation_switches(use_residual, spatial_encoder, kernels):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = Batch.from_data_list(
        [Data(x=torch.randn(3, 12), edge_index=edge_index, y=torch.tensor(0.0))]
    )
    model = MSResSTGAT(
        hidden_dim=16,
        heads=4,
        kernels=kernels,
        dropout=0.0,
        use_residual=use_residual,
        spatial_encoder=spatial_encoder,
    )
    assert model(batch)["logits"].shape == (1,)


def test_usad_losses_scores_and_backward():
    model = USADModel(input_dim=24, latent_dim=4)
    windows = torch.rand(3, 24)
    loss1, loss2 = model.losses(windows, epoch=2)
    loss1.backward(retain_graph=True)
    model.zero_grad(set_to_none=True)
    loss2.backward()
    assert model.anomaly_score(windows).shape == (3,)


def test_usad_training_subset_excludes_attacks():
    class Dataset:
        @staticmethod
        def targets():
            return torch.tensor([0, 1, 0, 1]).numpy()

        def __len__(self):
            return 4

    assert normal_only_subset(Dataset()).indices == [0, 2]


def test_balanced_limiter_accepts_normal_only_nested_subset():
    class Dataset:
        @staticmethod
        def targets():
            return torch.tensor([0, 1, 0, 1]).numpy()

        def __len__(self):
            return 4

    normal = Subset(Dataset(), [0, 2])
    limited = limited_balanced_subset(normal, 1, 11)
    assert len(limited) == 1
