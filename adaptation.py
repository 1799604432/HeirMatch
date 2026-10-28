"""领域适配的数据增强与目标函数，不含训练编排或检索评测。"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def augment_normalized(images: torch.Tensor) -> torch.Tensor:
    """对 ImageNet 归一化图像执行原训练中的同位置外观增强。"""
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    raw = images * std + mean
    batch = len(images)
    brightness = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.72, 1.28)
    contrast = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.75, 1.25)
    channel_gain = torch.empty(batch, 3, 1, 1, device=images.device).uniform_(0.88, 1.12)
    spatial_mean = raw.mean(dim=(2, 3), keepdim=True)
    raw = (raw - spatial_mean) * contrast + spatial_mean
    raw = raw * brightness * channel_gain
    if random.random() < 0.35:
        raw = F.avg_pool2d(raw, kernel_size=3, stride=1, padding=1)
    noise = torch.randn_like(raw) * torch.empty(
        batch, 1, 1, 1, device=images.device
    ).uniform_(0.0, 0.025)
    raw = (raw + noise).clamp(0.0, 1.0)
    return (raw - mean) / std


def domain_adaptation_loss(
    descriptors: torch.Tensor,
    patches: torch.Tensor,
    targets: torch.Tensor,
    *,
    retention_weight: float = 0.50,
    patch_weight: float = 0.25,
    variance_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    """原领域适配目标；前半批为增强 A，后半批为增强 B。

    descriptors=[2B,8448] 为 L2 归一化描述子，
    patches=[2B,768,H,W]、targets=[B,8448]。
    targets 为未微调 MegaLoc 的全局描述子；返回未做梯度累积缩放的
    总 loss 与各分项，调用方负责 backward 和优化器。
    方差阈值 0.02 是软惩罚目标；对单位归一化的 8448 维描述子，
    它不是所有维度都能同时满足的标准差下界。
    """
    if descriptors.ndim != 2 or descriptors.shape[0] != 2 * targets.shape[0]:
        raise ValueError("descriptors must contain two views per target")
    if patches.ndim != 4 or patches.shape[0] != descriptors.shape[0]:
        raise ValueError("patches must have shape [2*batch,channels,height,width]")
    targets = F.normalize(targets, dim=-1)
    descriptor_a, descriptor_b = descriptors.chunk(2)
    patch_a, patch_b = patches.chunk(2)
    invariance = (1.0 - F.cosine_similarity(descriptor_a, descriptor_b, dim=-1)).mean()
    retention = 0.5 * (
        (1.0 - F.cosine_similarity(descriptor_a, targets, dim=-1)).mean()
        + (1.0 - F.cosine_similarity(descriptor_b, targets, dim=-1)).mean()
    )
    patch_a = F.normalize(patch_a.flatten(2).transpose(1, 2), dim=-1)
    patch_b = F.normalize(patch_b.flatten(2).transpose(1, 2), dim=-1)
    patch_consistency = (1.0 - (patch_a * patch_b).sum(dim=-1)).mean()
    std = torch.sqrt(descriptors.float().var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(0.02 - std).mean()
    loss = (
        invariance + retention_weight * retention
        + patch_weight * patch_consistency + variance_weight * variance
    )
    return {
        "loss": loss,
        "invariance": invariance,
        "retention": retention,
        "patch": patch_consistency,
        "variance": variance,
    }
