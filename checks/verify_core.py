"""Compare the extracted core with the original algorithms on deterministic CPU inputs.

Run from any directory with ``python verify_core.py --project-root PROJECT``.
The original research scripts are loaded only for verification; the core
package itself does not import them. No dataset, training run, or GPU is needed.
An existing trained graph checkpoint is also checked when present.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F


def load_reference(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def close(actual, expected, *, atol=1e-7, rtol=1e-6) -> float:
    """Assert recursive output equality and return the largest absolute error."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), "Output dictionary keys differ"
        return max((close(actual[k], value, atol=atol, rtol=rtol)
                    for k, value in expected.items()), default=0.0)
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), "Output sequence lengths differ"
        return max((close(a, b, atol=atol, rtol=rtol)
                    for a, b in zip(actual, expected)), default=0.0)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return float((actual.detach() - expected.detach()).abs().max()) if actual.numel() else 0.0
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    core = Path(__file__).resolve().parents[1]
    scripts = project / "代码" / "dinov2-main" / "scripts"
    original_mega = scripts / "davi_megaloc"
    # These search-path changes are confined to this verification process.
    sys.path[:0] = [str(core), str(original_mega), str(scripts / "davi")]
    torch.set_num_threads(2)
    torch.manual_seed(20261028)
    rng = np.random.default_rng(20261028)
    started = perf_counter()
    records = []

    def check(name, function):
        begin = perf_counter()
        try:
            detail = function()
            status = "skipped" if isinstance(detail, dict) and detail.get("skipped") else "passed"
            record = {"name": name, "status": status, "details": detail}
        except Exception as error:
            record = {"name": name, "status": "failed", "error": str(error),
                      "traceback": traceback.format_exc()}
        record["seconds"] = round(perf_counter() - begin, 4)
        records.append(record)
        print(f"{record['status']}: {name}", flush=True)

    from herimatch import descriptors, losses, matching, nodes

    old_davi = load_reference("davi_model", scripts / "davi" / "davi_model.py")
    old_match = load_reference("_reference_multilevel", original_mega / "multilevel_graph_model.py")
    old_nodes = load_reference("_reference_nodes", original_mega / "extract_multilevel_graphs.py")
    old_global = load_reference("_reference_global", original_mega / "extract_tuned_megaloc.py")
    old_hspa = load_reference("_reference_hspa", original_mega / "build_hspa_descriptors.py")
    old_fusion = load_reference("_reference_fusion", original_mega / "prepare_multilevel_fusion.py")

    batch, levels, count, channels = 4, 4, 64, 256
    features_a = F.normalize(torch.randn(batch, levels, count, channels), dim=-1)
    features_b = F.normalize(features_a.roll(1, dims=2) + 0.02 * torch.randn_like(features_a), dim=-1)
    positions_a = torch.rand(batch, levels, count, 2)
    positions_b = torch.rand(batch, levels, count, 2)
    saliency_a = torch.rand(batch, levels, count)
    saliency_b = torch.rand(batch, levels, count)
    base = torch.tensor([0.91, 0.86, 0.95, 0.88])
    inputs = (features_a, positions_a, saliency_a, features_b, positions_b, saliency_b, base)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    groups = torch.tensor([0, 0, 1, 1])
    models = {}

    def parameters(encoder):
        old = old_match.MultiLevelDAVI(encoder=encoder).eval()
        # Nonzero head weights exercise both complete heads, rather than only
        # the zero-initialized baseline score of a newly created model.
        with torch.no_grad():
            for head in (old.classification_head, old.ranking_head):
                head[-1].weight.normal_(std=0.03)
                head[-1].bias.normal_(std=0.03)
        new = matching.MultiLevelDAVI(encoder=encoder).eval()
        assert old.state_dict().keys() == new.state_dict().keys()
        old_count = sum(p.numel() for p in old.parameters())
        new_count = sum(p.numel() for p in new.parameters())
        assert old_count == new_count
        if encoder == "gat":
            assert new_count == 1067734
        new.load_state_dict(old.state_dict(), strict=True)
        close(new.state_dict(), old.state_dict(), atol=0, rtol=0)
        models[encoder] = (old, new)
        return {"parameters": new_count, "state_entries": len(new.state_dict())}

    def forward(encoder):
        old, new = models[encoder]
        with torch.no_grad():
            expected = old(*inputs)
            actual = new(*inputs)
        assert len(actual["matches"]) == 4
        assert all(value.shape == (batch, count, count) for value in actual["matches"])
        return {"max_absolute_error": close(actual, expected), "matched_levels": 4}

    def objectives_and_gradient(encoder):
        old, new = models[encoder]
        old.zero_grad(set_to_none=True)
        new.zero_grad(set_to_none=True)
        old_output, new_output = old(*inputs), new(*inputs)
        pos_weight = torch.tensor(1.3)
        classification = F.binary_cross_entropy_with_logits(
            old_output["pair_logit"], labels, pos_weight=pos_weight
        )
        rank = old_davi.ranking_loss(old_output["retrieval_score"], labels, groups, 0.01)
        correspondence = old_match.multilevel_correspondence_loss(
            old_output["matches"], labels, features_a, features_b
        )
        regularization = old_output["residual"].pow(2).mean() / 0.05**2
        total = classification + 2.0 * rank + 0.1 * correspondence + 0.02 * regularization
        expected = {"loss": total, "classification": classification, "ranking": rank,
                    "matching": correspondence, "residual_regularization": regularization}
        actual = losses.graph_matching_loss(
            new_output, labels, groups, features_a, features_b, pos_weight=pos_weight
        )
        loss_error = close(actual, expected)
        total.backward()
        actual["loss"].backward()
        gradient_error, checked = 0.0, 0
        for (old_name, old_p), (new_name, new_p) in zip(old.named_parameters(), new.named_parameters()):
            assert old_name == new_name
            assert (old_p.grad is None) == (new_p.grad is None), old_name
            if old_p.grad is not None:
                assert torch.isfinite(new_p.grad).all(), new_name
                gradient_error = max(gradient_error, close(new_p.grad, old_p.grad))
                checked += 1
        assert checked > 0
        assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in new.parameters())
        return {"loss_max_absolute_error": loss_error, "gradient_max_absolute_error": gradient_error,
                "parameters_with_gradient": checked, "matching_loss": float(correspondence.detach())}

    for encoder in ("gat", "gcn"):
        check(f"{encoder}_parameters_and_strict_load", lambda encoder=encoder: parameters(encoder))
        check(f"{encoder}_all_forward_outputs", lambda encoder=encoder: forward(encoder))
        check(f"{encoder}_loss_terms_and_gradients", lambda encoder=encoder: objectives_and_gradient(encoder))

    def trained_checkpoint():
        path = project / "output" / "davi_megaloc" / "multilevel_gnn_gat1" / "best_multilevel_gnn.pth"
        if not path.is_file():
            return {"skipped": True, "reason": "Existing trained checkpoint is absent", "path": str(path)}
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        names = ("hidden_dim", "graph_layers", "heads", "neighbours", "encoder", "max_residual")
        options = {name: config[name] for name in names if name in config}
        old = old_match.MultiLevelDAVI(**options).eval()
        new = matching.MultiLevelDAVI(**options).eval()
        old.load_state_dict(checkpoint["model_state"], strict=True)
        new.load_state_dict(checkpoint["model_state"], strict=True)
        with torch.no_grad():
            error = close(new(*inputs), old(*inputs))
        return {"path": str(path), "max_absolute_error": error,
                "parameters": sum(p.numel() for p in new.parameters())}

    check("existing_trained_checkpoint", trained_checkpoint)

    patches = torch.randn(768, 23, 23)
    projected = torch.randn(256, 23, 23)
    tokens = patches.flatten(1).transpose(0, 1)
    reference_scores = old_nodes.scores(tokens, 23, 23)

    def score_comparison():
        error = close(nodes.scores(tokens, 23, 23), reference_scores)
        error = max(error, close(nodes.node_saliency(tokens, 23, 23),
                                 old_global.node_saliency(tokens, 23, 23)))
        return {"max_absolute_error": error, "grid": [23, 23]}

    check("patch_saliency_scores", score_comparison)

    def expected_graph(indices, source, saliency):
        indices = torch.as_tensor(indices, dtype=torch.long)
        y = torch.div(indices, 23, rounding_mode="floor").float()
        x = (indices % 23).float()
        return {"features": F.normalize(source[indices], dim=-1),
                "positions": torch.stack(((x + 0.5) / 23, (y + 0.5) / 23), dim=-1),
                "saliency": saliency[indices], "patch_indices": indices}

    def global_nodes():
        saliency = old_global.node_saliency(tokens, 23, 23)
        selected = old_global.select_nodes(saliency, 23, 23)
        close(nodes.select_nodes(saliency, 23, 23), selected, atol=0, rtol=0)
        error = close(nodes.build_global_nodes(patches), expected_graph(selected, tokens, saliency))
        return {"max_absolute_error": error, "nodes": len(selected), "feature_dimension": 768}

    check("global_node_indices_and_construction", global_nodes)

    def multilevel_nodes():
        selections = old_nodes.level_indices(*reference_scores, 23, 23)
        assert nodes.level_indices(*reference_scores, 23, 23) == selections
        source = projected.flatten(1).transpose(0, 1)
        level_scores = (*reference_scores, reference_scores[0])
        branches = [expected_graph(index, source, score)
                    for index, score in zip(selections, level_scores)]
        expected = {key: torch.stack([branch[key] for branch in branches]) for key in branches[0]}
        error = close(nodes.build_multilevel_nodes(patches, projected), expected)
        return {"max_absolute_error": error, "shape": [4, 64, 256]}

    check("four_level_indices_and_construction", multilevel_nodes)

    graph = nodes.build_global_nodes(patches)
    cached_features = graph["features"].numpy().astype(np.float16)
    cached_positions = graph["positions"].numpy().astype(np.float32)
    cached_saliency = graph["saliency"].numpy().astype(np.float16)

    def hspa_descriptors():
        error = 0.0
        for temperature in (0.0, 2.0, 5.0):
            expected = old_hspa.descriptor(cached_features, cached_positions, cached_saliency, temperature)
            actual = descriptors.descriptor(cached_features, cached_positions, cached_saliency, temperature)
            error = max(error, close(actual, expected))
            assert actual.shape == (8448,)
        close(descriptors.descriptor(cached_features, cached_positions, cached_saliency),
              old_hspa.descriptor(cached_features, cached_positions, cached_saliency, 2.0))
        return {"max_absolute_error": error, "temperatures": [0.0, 2.0, 5.0], "dimension": 8448}

    check("hspa_cached_node_descriptors", hspa_descriptors)

    def empty_regions():
        positions = np.full((64, 2), 0.1, dtype=np.float32)
        error = 0.0
        for features in (cached_features, np.zeros_like(cached_features)):
            expected = old_hspa.descriptor(features, positions, cached_saliency, 2.0)
            actual = descriptors.descriptor(features, positions, cached_saliency, 2.0)
            assert np.isfinite(actual).all()
            error = max(error, close(actual, expected))
        return {"max_absolute_error": error, "empty_region_fallback": True, "zero_features": True}

    check("hspa_empty_region_and_zero_feature_fallback", empty_regions)

    def fused_descriptors():
        mega = rng.normal(size=(5, 8448)).astype(np.float16)
        hspa = rng.normal(size=(5, 8448)).astype(np.float16)
        normalized_mega, normalized_hspa = old_fusion.normalize(mega), old_fusion.normalize(hspa)
        expected = np.concatenate((np.sqrt(0.6) * normalized_mega, np.sqrt(0.4) * normalized_hspa), axis=1)
        expected /= np.clip(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12, None)
        actual = descriptors.fuse_descriptors(mega, hspa)
        error = close(actual, expected)
        close(descriptors.fuse_descriptors(mega[0], hspa[0]), expected[0])
        cosine = 0.6 * (normalized_mega @ normalized_mega.T) + 0.4 * (normalized_hspa @ normalized_hspa.T)
        identity_error = close(actual @ actual.T, cosine, atol=2e-6, rtol=2e-6)
        return {"max_absolute_error": error, "weighted_cosine_max_absolute_error": identity_error,
                "dimension": actual.shape[1], "weights": [0.6, 0.4]}

    check("descriptor_fusion_and_weighted_cosine_identity", fused_descriptors)

    def boundary_cases():
        tied = torch.zeros(23 * 23)
        close(nodes.select_nodes(tied, 23, 23), old_global.select_nodes(tied, 23, 23), atol=0, rtol=0)
        mask = torch.zeros_like(tied, dtype=torch.bool)
        mask[:3] = True
        for selection in (nodes.select_from_mask, old_nodes.select_from_mask):
            try:
                selection(tied, mask, 64, 23)
            except RuntimeError:
                pass
            else:
                raise AssertionError("An insufficient node region must fail")
        negative_labels = torch.zeros(batch)
        assert losses.ranking_loss(base, negative_labels, groups).item() == 0.0
        probability = torch.ones(batch, count, count) / count
        assert losses.pseudo_correspondence_loss(
            probability, negative_labels, features_a[:, 0], features_b[:, 0]
        ).item() == 0.0
        for invalid in (0.0, -0.05):
            try:
                losses.graph_matching_loss({}, labels, groups, features_a, features_b, max_residual=invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Nonpositive max_residual must fail")
        return {"tie_order": True, "insufficient_region": True, "no_positive_pairs": True,
                "nonpositive_residual_bound": True}

    check("selection_and_loss_boundary_cases", boundary_cases)

    report = {"passed": sum(item["status"] == "passed" for item in records),
              "failed": sum(item["status"] == "failed" for item in records),
              "skipped": sum(item["status"] == "skipped" for item in records),
              "checks": records, "project_root": str(project), "device": "cpu",
              "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
              "numpy_version": np.__version__, "elapsed_seconds": round(perf_counter() - started, 4)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
