"""用现有两张图像缓存及权重检查编码入口；不运行实验或改原文件。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from herimatch.adaptation import augment_normalized, domain_adaptation_loss
from herimatch.backbone import (
    apply_trainable_delta, load_official_megaloc, preprocess_image,
)
from herimatch.pipeline import HeriMatch, load_matching_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=CORE.parent)
    parser.add_argument("--report", type=Path, default=CORE / "入口验证结果.json")
    args = parser.parse_args()
    project = args.project_root.resolve()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = project / "output" / "davi_megaloc"
    model = load_official_megaloc(device)
    apply_trainable_delta(model, root / "finetune" / "best_megaloc_delta.pth")
    matcher, checkpoint = load_matching_checkpoint(
        root / "multilevel_gnn_gat1" / "best_multilevel_gnn.pth", device
    )
    algorithm = HeriMatch(model, matcher)
    cache = np.load(project / "output/external_vpr/image_cache_322.f16.npy", mmap_mode="r")
    images = torch.from_numpy(np.array(cache[:2], dtype=np.float32)).to(device)
    actual = algorithm.encode(images)
    results = {}

    def compare(name, value, expected, atol):
        value = np.asarray(value)
        expected = np.asarray(expected)
        error = float(np.max(np.abs(value.astype(float) - expected.astype(float))))
        np.testing.assert_allclose(value, expected, rtol=0, atol=atol, err_msg=name)
        results[name] = {"passed": True, "max_absolute_error": error, "atol": atol}

    # 用同一模型、同一批输入执行原脚本的纯函数，隔离代码整理本身的影响。
    # 原缓存由批量12生成；直接拿它与批量2的GPU结果比较会混入矩阵核差异。
    scripts = project / "代码/dinov2-main/scripts/davi_megaloc"

    def load_legacy(name, filename):
        spec = importlib.util.spec_from_file_location(name, scripts / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    original_global = load_legacy("original_global_extraction", "extract_tuned_megaloc.py")
    original_graph = load_legacy("original_graph_extraction", "extract_multilevel_graphs.py")
    original_hspa = load_legacy("original_hspa", "build_hspa_descriptors.py")
    with torch.inference_mode():
        # extract_tuned_megaloc.py：全局描述子和原始768维节点。
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            global_descriptor, global_patches, _ = original_global.forward_megaloc_parts(
                model, images
            )
        global_values = global_descriptor.float().cpu().numpy().astype(np.float16)
        global_patches = global_patches.float()
        height, width = global_patches.shape[-2:]
        flat = global_patches.flatten(2).transpose(1, 2)
        hspa_values = []
        for tokens in flat:
            saliency = original_global.node_saliency(tokens, height, width)
            selected = original_global.select_nodes(saliency, height, width)
            y = torch.div(selected, width, rounding_mode="floor").float()
            x = (selected % width).float()
            positions = torch.stack(((x + .5) / width, (y + .5) / height), dim=1)
            # 与NPZ存储/读取一致：features=float16，positions=float32，
            # saliency=float16存储后转float32，再进行HSPA池化。
            features = F.normalize(tokens[selected], dim=-1).cpu().numpy().astype(np.float16)
            positions = positions.cpu().numpy().astype(np.float32)
            saliency = saliency[selected].cpu().numpy().astype(np.float16).astype(np.float32)
            hspa_values.append(original_hspa.descriptor(features, positions, saliency, 2.0))
        hspa_values = np.stack(hspa_values).astype(np.float16)

        # prepare_multilevel_fusion.py：分别归一化，再按平方根权重拼接。
        mega = global_values.astype(np.float32)
        mega /= np.clip(np.linalg.norm(mega, axis=1, keepdims=True), 1e-12, None)
        hspa = hspa_values.astype(np.float32)
        hspa /= np.clip(np.linalg.norm(hspa, axis=1, keepdims=True), 1e-12, None)
        fused = np.concatenate((np.sqrt(.6) * mega, np.sqrt(.4) * hspa), axis=1)
        fused /= np.clip(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12, None)
        reference = {
            "global_descriptor": global_values.astype(np.float32),
            "hspa_descriptor": hspa_values.astype(np.float32),
            "descriptor": fused.astype(np.float16).astype(np.float32),
        }

        # extract_multilevel_graphs.py 独立重新运行主干；eval模式下使用同一批输入。
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            patches, _ = model.backbone(images)
            projected = model.aggregator.agg.cluster_features(patches)
        patches, projected = patches.float(), projected.float()
        height, width = patches.shape[-2:]
        flat = patches.flatten(2).transpose(1, 2)
        projected_flat = projected.flatten(2).transpose(1, 2)
        graph_values = {key: [] for key in ("features", "positions", "saliency", "patch_indices")}
        for tokens, projected_tokens in zip(flat, projected_flat):
            base_score, horizontal, vertical = original_graph.scores(tokens, height, width)
            selections = original_graph.level_indices(
                base_score, horizontal, vertical, height, width
            )
            level_scores = (base_score, horizontal, vertical, base_score)
            per_image = {key: [] for key in graph_values}
            for selection, score in zip(selections, level_scores):
                selected = torch.tensor(selection, dtype=torch.long, device=device)
                y = torch.div(selected, width, rounding_mode="floor").float()
                x = (selected % width).float()
                positions = torch.stack(((x + .5) / width, (y + .5) / height), dim=1)
                per_image["features"].append(
                    F.normalize(projected_tokens[selected], dim=-1).cpu().numpy().astype(np.float16)
                )
                per_image["positions"].append(positions.cpu().numpy().astype(np.float16))
                per_image["saliency"].append(score[selected].cpu().numpy().astype(np.float16))
                per_image["patch_indices"].append(selected.cpu().numpy().astype(np.int16))
            for key in graph_values:
                graph_values[key].append(np.stack(per_image[key]))
        for key, values in graph_values.items():
            values = np.stack(values)
            reference[key] = values if key == "patch_indices" else values.astype(np.float32)

    for key, expected in reference.items():
        tolerance = 0.0 if key == "patch_indices" else 1e-6
        compare(key, actual[key].cpu().numpy(), expected, tolerance)

    expected_paths = {
        "global_descriptor": root / "features/cls_features.f16.npy",
        "hspa_descriptor": root / "hspa/hspa_temp_2p0.f16.npy",
        "descriptor": root / "multilevel_features/cls_features.f16.npy",
        "features": root / "multilevel_graphs/features.f16.npy",
        "positions": root / "multilevel_graphs/positions.f16.npy",
        "saliency": root / "multilevel_graphs/saliency.f16.npy",
        "patch_indices": root / "multilevel_graphs/patch_indices.i16.npy",
    }
    # 历史磁盘缓存仅用于诊断，不作为此次代码回归正确性的断言。
    cache_diagnostics = {}
    for key, path in expected_paths.items():
        expected = np.load(path, mmap_mode="r")[:2]
        value = actual[key].cpu().numpy()
        cache_diagnostics[key] = {
            "max_absolute_error": float(np.max(np.abs(value.astype(float) - expected.astype(float))))
        }
        if key == "patch_indices":
            cache_diagnostics[key]["ordered_node_agreement"] = float(np.mean(value == expected))
            cache_diagnostics[key]["selected_node_agreement"] = float(np.mean([
                len(set(current.tolist()) & set(old.tolist())) / len(current)
                for current, old in zip(value.reshape(-1, 64), expected.reshape(-1, 64))
            ]))

    paired = {key: value.flip(0) for key, value in actual.items()}
    scored = algorithm.match(actual, paired)
    base = (F.normalize(actual["descriptor"], dim=-1)
            * F.normalize(paired["descriptor"], dim=-1)).sum(-1)
    torch.testing.assert_close(scored["score"], base, rtol=0, atol=0)
    for alpha in (0.0, 1.5):
        output = algorithm.match(actual, paired, alpha=alpha)
        expected_score = base + alpha * output["residual"]
        with torch.no_grad():
            expected_probability = (
                output["calibration_scale"] * (expected_score - output["decision_center"])
                + matcher.classification_gate_logit.sigmoid() * output["graph_logit"]
            ).sigmoid()
        torch.testing.assert_close(output["score"], expected_score, rtol=0, atol=0)
        torch.testing.assert_close(output["probability"], expected_probability, rtol=0, atol=0)
    results["alpha_and_probability"] = {"passed": True, "alphas": [0.0, 1.5]}

    image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (97, 141, 3), dtype=np.uint8))
    expected = T.Compose([
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.Resize([322, 322], antialias=True),
    ])(image).half().float()
    torch.testing.assert_close(preprocess_image(image), expected, rtol=0, atol=0)
    results["preprocessing"] = {"passed": True, "bitwise_equal": True}

    path = project / "代码/dinov2-main/scripts/davi_megaloc/finetune_megaloc.py"
    spec = importlib.util.spec_from_file_location("original_adaptation", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    random.seed(8)
    torch.manual_seed(8)
    expected = legacy.augment_normalized(images)
    random.seed(8)
    torch.manual_seed(8)
    torch.testing.assert_close(augment_normalized(images), expected, rtol=0, atol=0)
    results["augmentation"] = {"passed": True, "bitwise_equal": True}

    torch.manual_seed(4)
    desc = F.normalize(torch.randn(4, 48), dim=-1).requires_grad_()
    patches = torch.randn(4, 12, 5, 5, requires_grad=True)
    targets = F.normalize(torch.randn(2, 48), dim=-1)
    loss = domain_adaptation_loss(desc, patches, targets)
    a, b = desc.chunk(2)
    pa, pb = patches.chunk(2)
    pa = F.normalize(pa.flatten(2).transpose(1, 2), dim=-1)
    pb = F.normalize(pb.flatten(2).transpose(1, 2), dim=-1)
    expected = (
        (1-F.cosine_similarity(a, b)).mean()
        + .5 * .5 * ((1-F.cosine_similarity(a, targets)).mean()
                     + (1-F.cosine_similarity(b, targets)).mean())
        + .25 * (1-(pa*pb).sum(-1)).mean()
        + .02 * F.relu(.02-torch.sqrt(desc.float().var(0, unbiased=False)+1e-4)).mean()
    )
    torch.testing.assert_close(loss["loss"], expected, rtol=0, atol=0)
    loss["loss"].backward()
    assert torch.isfinite(desc.grad).all() and torch.isfinite(patches.grad).all()
    results["adaptation_loss_and_gradients"] = {"passed": True}
    report = {
        "passed": True, "torch": torch.__version__, "numpy": np.__version__,
        "device": str(device), "images_checked": 2,
        "checkpoint_selected_alpha": checkpoint.get("selected_alpha"),
        "pipeline_default_alpha": algorithm.alpha,
        "regression_reference": "同一已加载模型、同一批2张真实图像，运行原始脚本函数并复刻原量化步骤",
        "historical_cache_diagnostics": {
            "used_for_correctness_assertions": False,
            "note": "历史缓存原提取批量为12，本检查使用批量2；GPU矩阵核的末位差异可能影响显著性排序边界，导致节点选择或顺序差异。代码回归以同批原始实现为准。",
            "comparisons": cache_diagnostics,
        },
        "checks": results,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
