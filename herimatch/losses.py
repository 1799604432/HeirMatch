"""Training objectives used by the four-level graph matcher.

Only differentiable algorithm components live here; dataset construction,
training loops, validation, and experiment reporting remain outside the core.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

def ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """Within-query positive-vs-hard-negative soft margin loss."""

    losses = []
    for group_id in group_ids.unique():
        selected = group_ids == group_id
        positives = scores[selected & (labels == 1)]
        negatives = scores[selected & (labels == 0)]
        if len(positives) and len(negatives):
            losses.append(F.softplus(margin - positives[:, None] + negatives[None, :]).mean())
    return torch.stack(losses).mean() if losses else scores.new_zeros(())


def pseudo_correspondence_loss(
    match_probability: torch.Tensor,
    labels: torch.Tensor,
    frozen_a: torch.Tensor,
    frozen_b: torch.Tensor,
    minimum_similarity: float = 0.30,
) -> torch.Tensor:
    """Align learned transport with mutual input-feature matches on positives."""

    losses = []
    with torch.no_grad():
        raw_a = F.normalize(frozen_a.float(), dim=-1)
        raw_b = F.normalize(frozen_b.float(), dim=-1)
        raw_similarity = torch.einsum("bkd,bld->bkl", raw_a, raw_b)
    for index in range(len(labels)):
        if int(labels[index].item()) != 1:
            continue
        similarity = raw_similarity[index]
        best_b = similarity.argmax(dim=1)
        best_a = similarity.argmax(dim=0)
        left = torch.arange(similarity.shape[0], device=similarity.device)
        valid = (best_a[best_b] == left) & (similarity[left, best_b] >= minimum_similarity)
        left, right = left[valid], best_b[valid]
        if len(left):
            row_probability = match_probability[index] / match_probability[index].sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-8)
            losses.append(-row_probability[left, right].clamp(min=1e-8).log().mean())
    return torch.stack(losses).mean() if losses else match_probability.new_zeros(())


def multilevel_correspondence_loss(matches, labels, features_a, features_b):
    """Average the positive-pair correspondence loss over all spatial levels."""

    losses = [
        pseudo_correspondence_loss(matches[level], labels, features_a[:, level], features_b[:, level])
        for level in range(features_a.shape[1])
    ]
    return torch.stack(losses).mean()


def graph_matching_loss(
    output,
    labels,
    groups,
    features_a,
    features_b,
    *,
    pos_weight=None,
    ranking_weight=2.0,
    matching_weight=0.1,
    residual_weight=0.02,
    max_residual=0.05,
    margin=0.01,
):
    """Combine the original classification, ranking, matching, and residual terms.

    ``output`` is the dictionary returned by ``MultiLevelDAVI``. ``groups``
    identifies the query of each labeled pair. ``pos_weight`` follows
    ``binary_cross_entropy_with_logits`` and should be a tensor on the same
    device as the logits when provided. Set ``max_residual`` to the value used
    to construct the model.
    """

    if max_residual <= 0:
        raise ValueError("max_residual must be positive for residual regularization")

    classification = F.binary_cross_entropy_with_logits(
        output["pair_logit"], labels, pos_weight=pos_weight
    )
    rank = ranking_loss(output["retrieval_score"], labels, groups, margin)
    matching = multilevel_correspondence_loss(
        output["matches"], labels, features_a, features_b
    )
    residual_reg = output["residual"].pow(2).mean() / max_residual**2
    loss = (
        classification
        + ranking_weight * rank
        + matching_weight * matching
        + residual_weight * residual_reg
    )
    return {
        "loss": loss,
        "classification": classification,
        "ranking": rank,
        "matching": matching,
        "residual_regularization": residual_reg,
    }


