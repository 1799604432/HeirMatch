"""显著性选点与四层图构建，不包含缓存、文件路径或实验流程。

HSPA 使用原始 768 维 patch 的全局选点；图匹配使用相同 patch
计算显著性，再收集聚合器投影后的 256 维特征。坐标统一为 (x, y)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


LEVEL_NAMES = ("global", "horizontal_3", "vertical_3", "grid_2x2")
NODES_PER_LEVEL = 64


def _check_tokens(tokens: torch.Tensor, height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tokens.ndim != 2 or tokens.shape[0] != height * width:
        raise ValueError("tokens must have shape [height * width, channels]")


def normalize_score(score: torch.Tensor) -> torch.Tensor:
    """将非负显著性分数缩放到 [0, 1]，保留原实现的稳定项。"""
    score = score.clamp(min=0.0)
    return (score - score.min()) / (score.max() - score.min() + 1e-6)


def scores(
    tokens: torch.Tensor, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回基础、水平带和垂直带选点分数，输入为 [H*W, C]。"""
    _check_tokens(tokens, height, width)
    descriptors = F.normalize(tokens, dim=-1)
    feature_map = (
        descriptors.reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0)
    )
    local_mean = F.avg_pool2d(feature_map, kernel_size=3, stride=1, padding=1)
    local = 1.0 - (
        feature_map * F.normalize(local_mean, dim=1)
    ).sum(dim=1).flatten()
    global_mean = F.normalize(descriptors.mean(dim=0, keepdim=True), dim=-1)
    global_score = 1.0 - (descriptors * global_mean).sum(dim=-1)

    left_right = 0.5 * (
        torch.roll(feature_map, 1, 3) + torch.roll(feature_map, -1, 3)
    )
    up_down = 0.5 * (
        torch.roll(feature_map, 1, 2) + torch.roll(feature_map, -1, 2)
    )
    diff_x = 1.0 - (
        feature_map * F.normalize(left_right, dim=1)
    ).sum(dim=1).flatten()
    diff_y = 1.0 - (
        feature_map * F.normalize(up_down, dim=1)
    ).sum(dim=1).flatten()
    base = 0.65 * normalize_score(local) + 0.35 * normalize_score(global_score)
    horizontal = 0.70 * base + 0.30 * normalize_score(diff_x)
    vertical = 0.70 * base + 0.30 * normalize_score(diff_y)
    return base, horizontal, vertical


def node_saliency(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """HSPA 原始选点显著性：0.65 局部差异 + 0.35 全局差异。"""
    _check_tokens(tokens, height, width)
    descriptors = F.normalize(tokens, dim=-1)
    feature_map = (
        descriptors.reshape(height, width, -1).permute(2, 0, 1).unsqueeze(0)
    )
    local_mean = F.avg_pool2d(feature_map, kernel_size=3, stride=1, padding=1)
    local = 1.0 - (
        feature_map * F.normalize(local_mean, dim=1)
    ).sum(dim=1).flatten()
    global_mean = F.normalize(descriptors.mean(dim=0, keepdim=True), dim=-1)
    global_score = 1.0 - (descriptors * global_mean).sum(dim=-1)
    return 0.65 * normalize_score(local) + 0.35 * normalize_score(global_score)


def select_from_mask(
    score: torch.Tensor,
    mask: torch.Tensor,
    count: int,
    width: int,
    minimum_distance: float = 2.0,
) -> list[int]:
    """按分数贪心选点；距离约束不足时按原分数顺序补足配额。"""
    if score.ndim != 1 or mask.shape != score.shape or mask.dtype != torch.bool:
        raise ValueError("score and boolean mask must have matching shape [H*W]")
    if count <= 0 or width <= 0 or minimum_distance < 0:
        raise ValueError("count/width must be positive and minimum_distance nonnegative")
    allowed = torch.nonzero(mask, as_tuple=False).flatten()
    ordered = allowed[torch.argsort(score[allowed], descending=True)].tolist()
    selected, coordinates = [], []
    for index in ordered:
        y, x = divmod(index, width)
        if all(
            (x - old_x) ** 2 + (y - old_y) ** 2 >= minimum_distance**2
            for old_x, old_y in coordinates
        ):
            selected.append(index)
            coordinates.append((x, y))
            if len(selected) == count:
                break
    used = set(selected)
    selected.extend(
        index for index in ordered if index not in used and len(selected) < count
    )
    if len(selected) != count:
        raise RuntimeError(f"selected={len(selected)} expected={count}")
    return selected


def select_nodes(
    score: torch.Tensor,
    height: int,
    width: int,
    count: int = 64,
    distance: float = 3.0,
) -> torch.Tensor:
    """全局选点，返回原 patch 展平顺序中的整数索引。"""
    if score.shape != (height * width,) or height <= 0 or width <= 0:
        raise ValueError("score must have shape [height * width] on a positive grid")
    # 保留全局选点原始 argsort 路径；分数相同时仍沿用 PyTorch 的排序行为。
    if count <= 0 or distance < 0:
        raise ValueError("count must be positive and distance nonnegative")
    ordered = torch.argsort(score, descending=True).tolist()
    selected, coordinates = [], []
    for index in ordered:
        y, x = divmod(index, width)
        if all(
            (x - old_x) ** 2 + (y - old_y) ** 2 >= distance**2
            for old_x, old_y in coordinates
        ):
            selected.append(index)
            coordinates.append((x, y))
            if len(selected) == count:
                break
    used = set(selected)
    selected.extend(
        index for index in ordered if index not in used and len(selected) < count
    )
    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} nodes, expected {count}")
    return torch.tensor(selected, dtype=torch.long, device=score.device)


def level_indices(
    base: torch.Tensor,
    horizontal: torch.Tensor,
    vertical: torch.Tensor,
    height: int,
    width: int,
) -> list[list[int]]:
    """每层独立选 64 点，顺序为全局、三水平带、三垂直带、2×2 网格。"""
    total = height * width
    if height <= 0 or width <= 0 or any(
        value.shape != (total,) for value in (base, horizontal, vertical)
    ):
        raise ValueError("scores must have shape [height * width] on a positive grid")
    all_mask = torch.ones(total, dtype=torch.bool, device=base.device)
    linear = torch.arange(total, device=base.device)
    y_coord = torch.div(linear, width, rounding_mode="floor")
    x_coord = linear % width
    global_nodes = select_from_mask(base, all_mask, 64, width, 3.0)

    horizontal_nodes, vertical_nodes = [], []
    for band, quota in enumerate((22, 21, 21)):
        mask_h = (y_coord >= band * height / 3) & (
            y_coord < (band + 1) * height / 3
        )
        horizontal_nodes.extend(
            select_from_mask(horizontal, mask_h, quota, width, 1.5)
        )
        mask_v = (x_coord >= band * width / 3) & (
            x_coord < (band + 1) * width / 3
        )
        vertical_nodes.extend(
            select_from_mask(vertical, mask_v, quota, width, 1.5)
        )

    grid_nodes = []
    for gy in range(2):
        for gx in range(2):
            mask = (
                (x_coord >= gx * width / 2)
                & (x_coord < (gx + 1) * width / 2)
                & (y_coord >= gy * height / 2)
                & (y_coord < (gy + 1) * height / 2)
            )
            grid_nodes.extend(select_from_mask(base, mask, 16, width, 1.5))
    return [global_nodes, horizontal_nodes, vertical_nodes, grid_nodes]


def _positions(selected: torch.Tensor, height: int, width: int) -> torch.Tensor:
    y = torch.div(selected, width, rounding_mode="floor").float()
    x = (selected % width).float()
    return torch.stack(((x + 0.5) / width, (y + 0.5) / height), dim=-1)


def build_global_nodes(patches: torch.Tensor) -> dict[str, torch.Tensor]:
    """从单图 [C,H,W] patch 构建 HSPA 所需的 64 个原始特征节点。

    输出 features=[64,C]、positions=[64,2]、saliency=[64]、
    patch_indices=[64]；浮点计算采用原提取脚本的 float32。
    """
    if patches.ndim != 3 or min(patches.shape) <= 0:
        raise ValueError("patches must have nonempty shape [channels, height, width]")
    height, width = patches.shape[-2:]
    tokens = patches.float().flatten(1).transpose(0, 1)
    saliency = node_saliency(tokens, height, width)
    selected = select_nodes(saliency, height, width)
    return {
        "features": F.normalize(tokens[selected], dim=-1),
        "positions": _positions(selected, height, width),
        "saliency": saliency[selected],
        "patch_indices": selected,
    }


def build_multilevel_nodes(
    patches: torch.Tensor, projected: torch.Tensor
) -> dict[str, torch.Tensor]:
    """以原始 patch 选点，收集投影特征供四层 GAT 使用。

    输入 patches=[C,H,W]、projected=[D,H,W]；输出 features=[4,64,D]、
    positions=[4,64,2]、saliency=[4,64]、patch_indices=[4,64]。
    """
    if patches.ndim != 3 or projected.ndim != 3:
        raise ValueError("patches and projected must have shape [channels, height, width]")
    if min(patches.shape) <= 0 or min(projected.shape) <= 0:
        raise ValueError("patches and projected must be nonempty")
    if patches.shape[-2:] != projected.shape[-2:] or patches.device != projected.device:
        raise ValueError("patches and projected must share their spatial grid and device")
    height, width = patches.shape[-2:]
    tokens = patches.float().flatten(1).transpose(0, 1)
    projected_tokens = projected.float().flatten(1).transpose(0, 1)
    base, horizontal, vertical = scores(tokens, height, width)
    selections = level_indices(base, horizontal, vertical, height, width)
    level_scores = (base, horizontal, vertical, base)
    selected = torch.tensor(selections, dtype=torch.long, device=patches.device)
    return {
        "features": F.normalize(projected_tokens[selected], dim=-1),
        "positions": _positions(selected, height, width),
        "saliency": torch.stack(
            [value[indices] for value, indices in zip(level_scores, selected)]
        ),
        "patch_indices": selected,
    }
