"""Spatial graph encoders and checkpoint-compatible four-level DAVI matching.

Inputs to ``MultiLevelDAVI`` have shapes [B,L,K,D] (features), [B,L,K,2]
(positions), [B,L,K] (saliency), and [B] (fused descriptor similarity).
Module attributes and operations retain the original model layout so its
``state_dict`` can be loaded with ``strict=True``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    multilevel_correspondence_loss,
    pseudo_correspondence_loss,
    ranking_loss,
)

def spatial_knn(positions: torch.Tensor, neighbours: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return spatial k-NN indices and distances for [B,K,2] positions."""

    distance = torch.cdist(positions, positions)
    nodes = positions.shape[1]
    k = min(neighbours + 1, nodes)
    indices = distance.topk(k=k, largest=False).indices[:, :, 1:]
    neighbour_distance = torch.gather(distance, 2, indices)
    return indices, neighbour_distance


class SpatialGATLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 4, neighbours: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden dimension must be divisible by the GAT head count")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.neighbours = neighbours
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.edge_bias = nn.Sequential(nn.Linear(3, heads), nn.Tanh())
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, nodes, dim = features.shape
        indices, distances = spatial_knn(positions, self.neighbours)
        neighbours = indices.shape[-1]
        expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, self.heads, self.head_dim
        )
        keys_all = self.k(features).view(batch, nodes, self.heads, self.head_dim)
        values_all = self.v(features).view(batch, nodes, self.heads, self.head_dim)
        keys = torch.gather(
            keys_all.unsqueeze(1).expand(-1, nodes, -1, -1, -1), 2, expanded
        )
        values = torch.gather(
            values_all.unsqueeze(1).expand(-1, nodes, -1, -1, -1), 2, expanded
        )
        query = self.q(features).view(batch, nodes, self.heads, self.head_dim)
        gathered_positions = torch.gather(
            positions.unsqueeze(1).expand(-1, nodes, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, 2),
        )
        delta = gathered_positions - positions.unsqueeze(2)
        edge = torch.cat((delta, distances.unsqueeze(-1)), dim=-1)
        logits = (query.unsqueeze(2) * keys).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits + self.edge_bias(edge)
        attention = logits.softmax(dim=2)
        message = (attention.unsqueeze(-1) * values).sum(dim=2).reshape(batch, nodes, dim)
        features = self.norm1(features + self.dropout(self.out(message)))
        return self.norm2(features + self.dropout(self.ffn(features)))


class SpatialGCNLayer(nn.Module):
    def __init__(self, dim: int, neighbours: int = 8, dropout: float = 0.1):
        super().__init__()
        self.neighbours = neighbours
        self.message = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = features.shape
        indices, distances = spatial_knn(positions, self.neighbours)
        adjacency = features.new_zeros((batch, nodes, nodes))
        sigma = distances.detach().mean(dim=(1, 2), keepdim=True).clamp(min=1e-3)
        weights = torch.exp(-(distances**2) / (2.0 * sigma**2))
        adjacency.scatter_(2, indices, weights)
        adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
        adjacency = adjacency + torch.eye(nodes, device=features.device).unsqueeze(0)
        degree = adjacency.sum(dim=-1).clamp(min=1e-6)
        normalizer = degree.rsqrt()
        adjacency = normalizer.unsqueeze(-1) * adjacency * normalizer.unsqueeze(-2)
        message = adjacency @ self.message(features)
        features = self.norm1(features + self.dropout(message))
        return self.norm2(features + self.dropout(self.ffn(features)))


class CrossGraphAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Reuse attention weights in both directions; retain the original
        # direction-specific normalization parameters for checkpoint compatibility.
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        update_a, _ = self.cross(a, b, b, need_weights=False)
        update_b, _ = self.cross(b, a, a, need_weights=False)
        return self.norm_a(a + self.dropout(update_a)), self.norm_b(b + self.dropout(update_b))


def sinkhorn_with_dustbin(
    similarity: torch.Tensor,
    dustbin_logit: torch.Tensor,
    temperature: torch.Tensor,
    iterations: int = 6,
) -> torch.Tensor:
    """Approximately doubly-normalise correspondences with unmatched nodes."""

    batch, rows, cols = similarity.shape
    logits = similarity / temperature.clamp(min=0.03, max=1.0)
    augmented = logits.new_empty((batch, rows + 1, cols + 1))
    augmented[:, :rows, :cols] = logits
    augmented[:, rows, :] = dustbin_logit
    augmented[:, :, cols] = dustbin_logit
    log_probability = augmented
    for _ in range(iterations):
        log_probability = log_probability - torch.logsumexp(log_probability, dim=2, keepdim=True)
        log_probability = log_probability - torch.logsumexp(log_probability, dim=1, keepdim=True)
    return log_probability.exp()[:, :rows, :cols]


class LevelBranch(nn.Module):
    """Encode and match one spatial level, returning eight match statistics."""

    def __init__(self, input_dim, hidden_dim, layers, heads, neighbours, encoder, dropout):
        super().__init__()
        self.feature = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()
        )
        self.position = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.saliency = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        factory = (
            (lambda: SpatialGATLayer(hidden_dim, heads, neighbours, dropout))
            if encoder == "gat" else
            (lambda: SpatialGCNLayer(hidden_dim, neighbours, dropout))
        )
        self.graph = nn.ModuleList([factory() for _ in range(layers)])
        self.cross = CrossGraphAttention(hidden_dim, heads, dropout)
        self.temperature = nn.Parameter(torch.tensor(0.10))
        self.dustbin = nn.Parameter(torch.tensor(0.0))

    def encode(self, features, positions, saliency):
        output = (
            self.feature(features)
            + self.position(positions)
            + self.saliency(saliency.unsqueeze(-1))
        )
        for layer in self.graph:
            output = layer(output, positions)
        return F.normalize(output, dim=-1)

    def forward(
        self, features_a, positions_a, saliency_a, features_b, positions_b, saliency_b
    ):
        a = self.encode(features_a, positions_a, saliency_a)
        b = self.encode(features_b, positions_b, saliency_b)
        a, b = self.cross(a, b)
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        similarity = torch.einsum("bkd,bld->bkl", a, b)
        match = sinkhorn_with_dustbin(similarity, self.dustbin, self.temperature, 6)
        row = match / match.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        col = match / match.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        row_best, col_best = row.max(dim=-1).values, col.max(dim=-2).values
        sim_row, sim_col = similarity.max(dim=-1).values, similarity.max(dim=-2).values
        expected_b = row @ positions_b
        expected_a = col.transpose(1, 2) @ positions_a
        da, db = expected_b - positions_a, expected_a - positions_b
        variance = 0.5 * (
            (da - da.mean(dim=1, keepdim=True)).pow(2).mean(dim=(1, 2))
            + (db - db.mean(dim=1, keepdim=True)).pow(2).mean(dim=(1, 2))
        )
        geometry = torch.exp(-10.0 * variance)
        entropy_row = (
            -(row.clamp(min=1e-8).log() * row).sum(dim=-1).mean(dim=-1)
            / math.log(row.shape[-1])
        )
        entropy_col = (
            -(col.clamp(min=1e-8).log() * col).sum(dim=-2).mean(dim=-1)
            / math.log(col.shape[-2])
        )
        top = min(8, row_best.shape[-1])
        stats = torch.stack(
            (
                match.sum(dim=(1, 2)) / match.shape[1],
                0.5 * (row_best.mean(-1) + col_best.mean(-1)),
                torch.minimum(row_best.mean(-1), col_best.mean(-1)),
                0.5 * (
                    row_best.topk(top, dim=-1).values.mean(-1)
                    + col_best.topk(top, dim=-1).values.mean(-1)
                ),
                0.5 * (sim_row.mean(-1) + sim_col.mean(-1)),
                0.5 * (
                    sim_row.topk(top, dim=-1).values.mean(-1)
                    + sim_col.topk(top, dim=-1).values.mean(-1)
                ),
                geometry,
                1.0 - 0.5 * (entropy_row + entropy_col),
            ),
            dim=-1,
        )
        return stats, match, similarity


class MultiLevelDAVI(nn.Module):
    """Four independent graph branches with dynamic level fusion and two heads."""

    def __init__(
        self,
        input_dim=256,
        levels=4,
        hidden_dim=128,
        graph_layers=1,
        heads=4,
        neighbours=8,
        encoder="gat",
        dropout=0.1,
        max_residual=0.05,
    ):
        super().__init__()
        self.levels = levels
        self.max_residual = max_residual
        self.branches = nn.ModuleList([
            LevelBranch(input_dim, hidden_dim, graph_layers, heads, neighbours, encoder, dropout)
            for _ in range(levels)
        ])
        summary_dim = levels * 8 + 1
        self.level_gate = nn.Sequential(nn.Linear(levels * 8, levels), nn.Softmax(dim=-1))
        self.classification_head = nn.Sequential(
            nn.LayerNorm(summary_dim), nn.Linear(summary_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.ranking_head = nn.Sequential(
            nn.LayerNorm(summary_dim), nn.Linear(summary_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.classification_head[-1].weight)
        nn.init.zeros_(self.classification_head[-1].bias)
        nn.init.zeros_(self.ranking_head[-1].weight)
        nn.init.zeros_(self.ranking_head[-1].bias)
        self.residual_gate_logit = nn.Parameter(torch.tensor(-1.0))
        self.classification_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.decision_center = nn.Parameter(torch.tensor(0.90))

    def forward(
        self, features_a, positions_a, saliency_a,
        features_b, positions_b, saliency_b, base_similarity
    ):
        stats, matches, similarities = [], [], []
        for level, branch in enumerate(self.branches):
            value, match, similarity = branch(
                features_a[:, level], positions_a[:, level], saliency_a[:, level],
                features_b[:, level], positions_b[:, level], saliency_b[:, level],
            )
            stats.append(value)
            matches.append(match)
            similarities.append(similarity)
        level_stats = torch.stack(stats, dim=1)
        attention = self.level_gate(level_stats.flatten(1))
        weighted = (attention.unsqueeze(-1) * level_stats).flatten(1)
        summary = torch.cat((weighted, base_similarity.unsqueeze(-1)), dim=-1)
        graph_logit = self.classification_head(summary).squeeze(-1)
        ranking_logit = self.ranking_head(summary).squeeze(-1)
        residual = (
            self.max_residual * torch.sigmoid(self.residual_gate_logit)
            * torch.tanh(ranking_logit)
        )
        score = base_similarity + residual
        scale = self.logit_scale.clamp(0.0, 5.0).exp()
        pair_logit = (
            scale * (score - self.decision_center)
            + torch.sigmoid(self.classification_gate_logit) * graph_logit
        )
        return {
            "pair_logit": pair_logit,
            "retrieval_score": score,
            "residual": residual,
            "graph_logit": graph_logit,
            "ranking_logit": ranking_logit,
            "matches": matches,
            "similarities": similarities,
            "level_attention": attention,
            "calibration_scale": scale,
            "decision_center": self.decision_center,
        }



