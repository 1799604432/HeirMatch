"""HSPA 空间金字塔池化与 MegaLoc 描述子融合。"""

from __future__ import annotations

import numpy as np


def normalized(vector: np.ndarray) -> np.ndarray:
    """对单个向量作 L2 归一化。"""
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def pool(features, positions, saliency, mask, temperature, fallback):
    """单区域显著性加权池化；空区域使用全局均值作为回退。

    positions 参数保留原函数接口，区域归属由 mask 给出。
    """
    selected = np.flatnonzero(mask)
    if len(selected) == 0:
        return fallback
    score = saliency[selected] * temperature
    score = np.exp(score - score.max())
    score /= np.clip(score.sum(), 1e-12, None)
    return normalized((score[:, None] * features[selected]).sum(axis=0))


def descriptor(
    features: np.ndarray,
    positions: np.ndarray,
    saliency: np.ndarray,
    temperature: float = 2.0,
) -> np.ndarray:
    """从 [K,C] 节点生成 [11*C] 的 HSPA 描述子。

    11 个池化单元依次为全局、3 水平带、3 垂直带和 2×2 网格。
    坐标 positions=[K,2] 按 (x,y) 给出；池化后执行有符号平方根
    和整体 L2 归一化。默认原始节点 C=768，输出维度为 8448。
    """
    features = np.asarray(features)
    positions = np.asarray(positions)
    saliency = np.asarray(saliency)
    if features.ndim != 2 or min(features.shape) <= 0:
        raise ValueError("features must have nonempty shape [nodes, channels]")
    if positions.shape != (len(features), 2) or saliency.shape != (len(features),):
        raise ValueError("positions must be [nodes,2] and saliency must be [nodes]")
    if not np.isfinite(temperature):
        raise ValueError("temperature must be finite")
    # astype 默认复制，防止原位归一化修改调用方输入。
    features = features.astype(np.float32)
    features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
    fallback = normalized(features.mean(axis=0))
    masks = [np.ones(len(features), dtype=bool)]
    for index in range(3):
        masks.append(
            (positions[:, 1] >= index / 3)
            & (positions[:, 1] < (index + 1) / 3)
        )
    for index in range(3):
        masks.append(
            (positions[:, 0] >= index / 3)
            & (positions[:, 0] < (index + 1) / 3)
        )
    for y in range(2):
        for x in range(2):
            masks.append(
                (positions[:, 0] >= x / 2)
                & (positions[:, 0] < (x + 1) / 2)
                & (positions[:, 1] >= y / 2)
                & (positions[:, 1] < (y + 1) / 2)
            )
    cells = [
        pool(features, positions, saliency, mask, temperature, fallback)
        for mask in masks
    ]
    output = np.concatenate(cells)
    # 降低夯土重复纹理造成的突发性。
    output = np.sign(output) * np.sqrt(np.abs(output))
    return normalized(output)


def fuse_descriptors(
    megaloc: np.ndarray,
    hspa: np.ndarray,
    megaloc_weight: float = 0.6,
) -> np.ndarray:
    """平方根加权拼接，默认余弦相似度为 0.6 MegaLoc + 0.4 HSPA。

    接受单个向量 [D] 或批量 [N,D]；两分支的批量大小必须一致。
    每个分支先转 float32 并 L2 归一化，拼接后再整体归一化；
    返回类型沿用 NumPy 对平方根权重乘法的类型提升规则。
    """
    megaloc = np.asarray(megaloc, dtype=np.float32)
    hspa = np.asarray(hspa, dtype=np.float32)
    if megaloc.ndim not in (1, 2) or hspa.ndim != megaloc.ndim:
        raise ValueError("descriptors must both be vectors [D] or batches [N,D]")
    if megaloc.shape[:-1] != hspa.shape[:-1]:
        raise ValueError("MegaLoc and HSPA must have matching batch dimensions")
    if megaloc.shape[-1] == 0 or hspa.shape[-1] == 0:
        raise ValueError("descriptor dimensions must be nonempty")
    if not np.isfinite(megaloc_weight) or not 0.0 <= megaloc_weight <= 1.0:
        raise ValueError("megaloc_weight must be finite and in [0,1]")
    megaloc = megaloc / np.clip(
        np.linalg.norm(megaloc, axis=-1, keepdims=True), 1e-12, None
    )
    hspa = hspa / np.clip(
        np.linalg.norm(hspa, axis=-1, keepdims=True), 1e-12, None
    )
    fused = np.concatenate(
        (np.sqrt(megaloc_weight) * megaloc, np.sqrt(1.0 - megaloc_weight) * hspa),
        axis=-1,
    )
    return fused / np.clip(
        np.linalg.norm(fused, axis=-1, keepdims=True), 1e-12, None
    )
