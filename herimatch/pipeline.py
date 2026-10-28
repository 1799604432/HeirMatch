"""图像编码与成对匹配接口，连接纯算法模块，不包含实验流程。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .backbone import forward_megaloc_parts
from .descriptors import descriptor, fuse_descriptors
from .matching import MultiLevelDAVI
from .nodes import build_global_nodes, build_multilevel_nodes


def load_matching_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[MultiLevelDAVI, dict]:
    """严格加载原 best_multilevel_gnn.pth，返回模型及检查点内容。

    selected_alpha 保留在返回的检查点中；全图库使用的 alpha 由调用方
    显式指定，不能直接沿用候选集验证参数。
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    keys = (
        "input_dim", "levels", "hidden_dim", "graph_layers", "heads",
        "neighbours", "encoder", "dropout", "max_residual",
    )
    model = MultiLevelDAVI(**{key: config[key] for key in keys if key in config})
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def score_graph_pairs(
    model: MultiLevelDAVI,
    graph_a: dict[str, torch.Tensor],
    graph_b: dict[str, torch.Tensor],
    base_similarity: torch.Tensor,
    *,
    alpha: float = 0.0,
) -> dict:
    """应用 alpha 后重算检索分数及匹配概率；默认当前全图库 alpha=0。

    图字段 features=[B,4,64,256]、positions=[B,4,64,2]、
    saliency=[B,4,64]；base_similarity=[B] 为融合描述子余弦相似度。
    原模型字段保持原样，另返回 score / probability / alpha。
    """
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    output = model(
        graph_a["features"], graph_a["positions"], graph_a["saliency"],
        graph_b["features"], graph_b["positions"], graph_b["saliency"],
        base_similarity,
    )
    score = base_similarity + alpha * output["residual"]
    logit = (
        output["calibration_scale"] * (score - output["decision_center"])
        + torch.sigmoid(model.classification_gate_logit) * output["graph_logit"]
    )
    return {**output, "score": score, "probability": logit.sigmoid(), "alpha": alpha}


class HeriMatch:
    """以现有 MegaLoc / MultiLevelDAVI 实例构建推理入口。

    各模型仍保留独立 state_dict。输入图像应先经 preprocess_image，
    encode 默认模拟原缓存的 float16 存储后 float32 读取过程。
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        matcher: MultiLevelDAVI,
        *,
        temperature: float = 2.0,
        megaloc_weight: float = 0.6,
        alpha: float = 0.0,
    ):
        self.backbone = backbone.eval()
        self.matcher = matcher.eval()
        self.temperature = temperature
        self.megaloc_weight = megaloc_weight
        self.alpha = alpha

    @torch.inference_mode()
    def encode(
        self, images: torch.Tensor, *, cache_precision: bool = True
    ) -> dict[str, torch.Tensor]:
        """[B,3,322,322] → 融合描述子及四层图；结果与输入在同一设备。"""
        if images.ndim != 4 or images.shape[1:] != (3, 322, 322) or len(images) == 0:
            raise ValueError("images must have nonempty shape [batch,3,322,322]")
        device = images.device
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            global_descriptors, patches, _ = forward_megaloc_parts(self.backbone, images)
            projected = self.backbone.aggregator.agg.cluster_features(patches)
        global_values = global_descriptors.float().cpu().numpy()
        if cache_precision:
            global_values = global_values.astype(np.float16)
        hspa_values, graphs = [], []
        for patch, projection in zip(patches, projected):
            nodes = build_global_nodes(patch)
            features = nodes["features"].cpu().numpy()
            saliency = nodes["saliency"].cpu().numpy()
            if cache_precision:
                features = features.astype(np.float16)
                saliency = saliency.astype(np.float16).astype(np.float32)
            hspa_values.append(descriptor(
                features, nodes["positions"].cpu().numpy(), saliency, self.temperature
            ))
            graphs.append(build_multilevel_nodes(patch, projection))
        hspa_values = np.stack(hspa_values)
        if cache_precision:
            hspa_values = hspa_values.astype(np.float16)
        fused = fuse_descriptors(global_values, hspa_values, self.megaloc_weight)
        if cache_precision:
            fused = fused.astype(np.float16)
        result = {
            "descriptor": torch.from_numpy(fused.astype(np.float32)).to(device),
            "global_descriptor": torch.from_numpy(global_values.astype(np.float32)).to(device),
            "hspa_descriptor": torch.from_numpy(hspa_values.astype(np.float32)).to(device),
        }
        for key in ("features", "positions", "saliency", "patch_indices"):
            value = torch.stack([graph[key] for graph in graphs])
            if cache_precision and value.is_floating_point():
                value = value.half().float()
            result[key] = value
        return result

    @torch.inference_mode()
    def match(self, encoded_a: dict, encoded_b: dict, *, alpha: float | None = None) -> dict:
        """等长批次逐对匹配；score 为校准后检索分数，probability 为匹配概率。"""
        if encoded_a["descriptor"].shape != encoded_b["descriptor"].shape:
            raise ValueError("encoded batches must have matching descriptor shapes")
        base = (
            F.normalize(encoded_a["descriptor"], dim=-1)
            * F.normalize(encoded_b["descriptor"], dim=-1)
        ).sum(dim=-1)
        return score_graph_pairs(
            self.matcher, encoded_a, encoded_b, base,
            alpha=self.alpha if alpha is None else alpha,
        )
