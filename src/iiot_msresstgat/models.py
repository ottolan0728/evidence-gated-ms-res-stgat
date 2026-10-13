from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_max_pool, global_mean_pool


POOLING_MODES = {"mean", "mean_max_std"}


def temporal_statistics(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Retain mean, extreme, and dispersion information over the last axis."""
    mean = values.mean(dim=-1)
    maximum = values.amax(dim=-1)
    variance = (values.square().mean(dim=-1) - mean.square()).clamp_min(0.0)
    standard_deviation = torch.sqrt(variance + epsilon)
    return torch.cat((mean, maximum, standard_deviation), dim=-1)


class MultiScaleTemporalEncoder(nn.Module):
    """Encode each sensor's time window before spatial message passing."""

    def __init__(
        self,
        hidden_dim: int,
        kernels: tuple[int, ...],
        dropout: float,
        pooling: str = "mean",
        pooling_epsilon: float = 1e-8,
    ):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"Unsupported temporal pooling: {pooling!r}")
        self.pooling = pooling
        self.pooling_epsilon = pooling_epsilon
        self.branches = nn.ModuleList(
            nn.Conv1d(1, hidden_dim, kernel, padding=kernel // 2) for kernel in kernels
        )
        multiplier = 1 if pooling == "mean" else 3
        self.projection = nn.Linear(hidden_dim * len(kernels) * multiplier, hidden_dim)
        if pooling == "mean_max_std":
            self.statistics_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_windows: torch.Tensor) -> torch.Tensor:
        sequences = node_windows.unsqueeze(1)
        activations = [F.gelu(layer(sequences)) for layer in self.branches]
        if self.pooling == "mean":
            branches = [values.mean(dim=-1) for values in activations]
            encoded = self.projection(torch.cat(branches, dim=-1))
        else:
            branches = [
                temporal_statistics(values, self.pooling_epsilon)
                for values in activations
            ]
            encoded = self.statistics_norm(
                F.gelu(self.projection(torch.cat(branches, dim=-1)))
            )
        return self.dropout(encoded)


class BinaryGraphModel(nn.Module):
    def _output(self, representation: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return {
            "logits": self.classifier(representation).squeeze(-1),
            "representation": representation,
            "attention": None,
        }


class MSResSTGAT(BinaryGraphModel):
    def __init__(
        self,
        hidden_dim: int = 64,
        heads: int = 4,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
        use_residual: bool = True,
        spatial_encoder: str = "gatv2",
        temporal_pooling: str = "mean",
        sensor_pooling: str = "mean",
        pooling_epsilon: float = 1e-8,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if spatial_encoder not in {"gatv2", "mean_pool"}:
            raise ValueError(f"Unsupported spatial encoder: {spatial_encoder!r}")
        if sensor_pooling not in POOLING_MODES:
            raise ValueError(f"Unsupported sensor pooling: {sensor_pooling!r}")
        self.use_residual = use_residual
        self.spatial_encoder = spatial_encoder
        self.sensor_pooling = sensor_pooling
        self.pooling_epsilon = pooling_epsilon
        self.temporal = MultiScaleTemporalEncoder(
            hidden_dim,
            kernels,
            dropout,
            pooling=temporal_pooling,
            pooling_epsilon=pooling_epsilon,
        )
        if spatial_encoder == "gatv2":
            self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        if sensor_pooling == "mean_max_std":
            self.sensor_projection = nn.Linear(hidden_dim * 3, hidden_dim)
            self.sensor_statistics_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, return_attention: bool = False):
        temporal = self.temporal(data.x)
        attention = None
        if self.spatial_encoder == "mean_pool":
            spatial = temporal
        else:
            spatial = F.elu(self.gat1(temporal, data.edge_index))
            spatial = self.norm1(spatial + temporal if self.use_residual else spatial)
            spatial_output = self.gat2(
                spatial, data.edge_index, return_attention_weights=return_attention
            )
            if return_attention:
                spatial, attention = spatial_output
            else:
                spatial = spatial_output
            spatial = F.elu(spatial)
            spatial = self.norm2(spatial + temporal if self.use_residual else spatial)
        mean = global_mean_pool(spatial, data.batch)
        if self.sensor_pooling == "mean":
            representation = mean
        else:
            maximum = global_max_pool(spatial, data.batch)
            mean_squared = global_mean_pool(spatial.square(), data.batch)
            variance = (mean_squared - mean.square()).clamp_min(0.0)
            standard_deviation = torch.sqrt(variance + self.pooling_epsilon)
            statistics = torch.cat((mean, maximum, standard_deviation), dim=-1)
            representation = self.sensor_statistics_norm(
                F.gelu(self.sensor_projection(statistics))
            )
        output = self._output(representation)
        output["attention"] = attention
        return output


class TemporalOnlyModel(BinaryGraphModel):
    """TCN baseline treating sensors as channels."""

    def __init__(self, num_nodes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = nn.Sequential(
            nn.Conv1d(num_nodes, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        windows = data.x.view(data.num_graphs, self.num_nodes, -1)
        representation = self.encoder(windows).mean(dim=-1)
        return self._output(representation)


class RecurrentOnlyModel(BinaryGraphModel):
    def __init__(self, kind: str, num_nodes: int, hidden_dim: int, dropout: float):
        super().__init__()
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.num_nodes = num_nodes
        self.encoder = recurrent(num_nodes, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        sequences = data.x.view(data.num_graphs, self.num_nodes, -1).transpose(1, 2)
        _, state = self.encoder(sequences)
        hidden = state[0] if isinstance(state, tuple) else state
        representation = self.dropout(hidden[-1])
        return self._output(representation)


class SpatialOnlyModel(BinaryGraphModel):
    def __init__(self, kind: str, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(1, hidden_dim)
        if kind == "gcn":
            self.layer1 = GCNConv(hidden_dim, hidden_dim)
            self.layer2 = GCNConv(hidden_dim, hidden_dim)
        else:
            if hidden_dim % heads:
                raise ValueError("hidden_dim must be divisible by heads")
            self.layer1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            self.layer2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        nodes = self.input_projection(data.x.mean(dim=-1, keepdim=True))
        nodes = self.dropout(F.gelu(self.layer1(nodes, data.edge_index)))
        nodes = F.gelu(self.layer2(nodes, data.edge_index))
        return self._output(global_mean_pool(nodes, data.batch))


class GCNGRUModel(BinaryGraphModel):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.temporal = nn.GRU(1, hidden_dim, batch_first=True)
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        _, hidden = self.temporal(data.x.unsqueeze(-1))
        nodes = self.dropout(F.gelu(self.gcn1(hidden[-1], data.edge_index)))
        nodes = F.gelu(self.gcn2(nodes, data.edge_index))
        return self._output(global_mean_pool(nodes, data.batch))


class USADModel(nn.Module):
    """Dual-decoder autoencoder from Audibert et al. (KDD 2020)."""

    def __init__(self, input_dim: int, latent_dim: int = 40):
        super().__init__()
        half, quarter = input_dim // 2, input_dim // 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, half),
            nn.ReLU(),
            nn.Linear(half, quarter),
            nn.ReLU(),
            nn.Linear(quarter, latent_dim),
            nn.ReLU(),
        )
        def decoder() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(latent_dim, quarter),
                nn.ReLU(),
                nn.Linear(quarter, half),
                nn.ReLU(),
                nn.Linear(half, input_dim),
                nn.Sigmoid(),
            )
        self.decoder1 = decoder()
        self.decoder2 = decoder()

    def reconstructions(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = self.decoder1(self.encoder(windows))
        adversarial = self.decoder2(self.encoder(first))
        return first, adversarial

    def losses(self, windows: torch.Tensor, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(windows)
        first = self.decoder1(encoded)
        second = self.decoder2(encoded)
        adversarial = self.decoder2(self.encoder(first))
        weight = 1.0 / epoch
        loss1 = weight * F.mse_loss(first, windows) + (1 - weight) * F.mse_loss(
            adversarial, windows
        )
        loss2 = weight * F.mse_loss(second, windows) - (1 - weight) * F.mse_loss(
            adversarial, windows
        )
        return loss1, loss2

    def anomaly_score(
        self, windows: torch.Tensor, alpha: float = 0.5, beta: float = 0.5
    ) -> torch.Tensor:
        first, adversarial = self.reconstructions(windows)
        return alpha * (windows - first).square().mean(dim=1) + beta * (
            windows - adversarial
        ).square().mean(dim=1)


def build_model(config: dict, num_nodes: int = 51) -> nn.Module:
    name = config.get("name", "ms_res_stgat")
    hidden_dim = int(config.get("hidden_dim", 64))
    heads = int(config.get("heads", 4))
    dropout = float(config.get("dropout", 0.1))
    if name == "ms_res_stgat":
        return MSResSTGAT(
            hidden_dim=hidden_dim,
            heads=heads,
            kernels=tuple(config.get("kernels", (3, 5, 7))),
            dropout=dropout,
            use_residual=bool(config.get("use_residual", True)),
            spatial_encoder=str(config.get("spatial_encoder", "gatv2")),
            temporal_pooling=str(config.get("temporal_pooling", "mean")),
            sensor_pooling=str(config.get("sensor_pooling", "mean")),
            pooling_epsilon=float(config.get("pooling_epsilon", 1e-8)),
        )
    if name == "tcn":
        return TemporalOnlyModel(num_nodes, hidden_dim, dropout)
    if name in {"gru", "lstm"}:
        return RecurrentOnlyModel(name, num_nodes, hidden_dim, dropout)
    if name in {"gcn", "gatv2"}:
        return SpatialOnlyModel(name, hidden_dim, heads, dropout)
    if name == "gcn_gru":
        return GCNGRUModel(hidden_dim, dropout)
    raise ValueError(f"Unknown neural model: {name!r}")
