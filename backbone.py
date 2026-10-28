"""MegaLoc 主干、参数高效适配及原项目增量权重接口。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


MEGALOC_REVISION = "a0f34722c4297ff787e022433799250180860af7"


def load_official_megaloc(
    device: torch.device | str = "cpu",
    *,
    weights_path: str | Path | None = None,
    revision: str = MEGALOC_REVISION,
) -> torch.nn.Module:
    """加载保留原样的 MegaLoc 实现及 safetensors 基础权重。

    显式提供 weights_path 可完全离线使用；否则从 Hugging Face 缓存或
    固定 revision 下载权重。加载不依赖原项目目录或 sys.path 修改。
    """
    from safetensors.torch import load_file

    from ._vendor.megaloc.megaloc_model import MegaLoc

    if weights_path is None:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(
            repo_id="gberton/MegaLoc",
            filename="model.safetensors",
            revision=revision,
        )
    model = MegaLoc()
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    return model.to(device).eval()


def configure_parameter_efficient_finetune(model: torch.nn.Module) -> list[str]:
    """按论文解冻最后一个 ViT block、最终 norm、OT 聚合及最终投影。

    最终线性投影为 aggregator.linear，独立于 aggregator.agg。
    调用方应在本函数之后，基于 requires_grad 参数构建优化器。
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (
        model.backbone.blocks[-1], model.backbone.norm,
        model.aggregator.agg, model.aggregator.linear,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [name for name, value in model.named_parameters() if value.requires_grad]


def save_trainable_delta(
    model: torch.nn.Module, path: str | Path, metadata: dict
) -> None:
    """沿用原项目 trainable_state + metadata 格式和 float16 存储。"""
    state = {
        name: parameter.detach().cpu().to(torch.float16)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable_state": state, "metadata": metadata}, path)


def apply_trainable_delta(model: torch.nn.Module, path: str | Path) -> dict:
    """将原项目增量叠加到同一基础模型，返回原 metadata。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    state = checkpoint["trainable_state"]
    missing = [name for name in state if name not in parameters]
    if missing:
        raise KeyError(f"Delta parameters not found in MegaLoc: {missing[:10]}")
    for name, value in state.items():
        if value.shape != parameters[name].shape:
            raise ValueError(f"Delta parameter shape mismatch: {name}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))
    return checkpoint.get("metadata", {})


def forward_megaloc_parts(model: torch.nn.Module, images: torch.Tensor):
    """返回归一化全局描述子、空间 patch map 和 CLS token。"""
    patches, cls_token = model.backbone(images)
    descriptor = model.l2norm(model.aggregator((patches, cls_token)))
    return F.normalize(descriptor, dim=-1), patches, cls_token


def preprocess_image(image, *, cache_precision: bool = True) -> torch.Tensor:
    """PIL 图像 → [3,322,322]，保持原缓存的变换顺序及默认量化。

    原流程为 ToTensor → ImageNet Normalize → Resize(antialias=True)，
    并以 float16 缓存后转回 float32；cache_precision=False 可跳过量化。
    """
    from torchvision import transforms as T

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.Resize([322, 322], antialias=True),
    ])
    tensor = transform(image.convert("RGB"))
    return tensor.half().float() if cache_precision else tensor
