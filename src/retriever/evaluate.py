"""Retrieval evaluation metrics: Recall@K and MRR."""

import torch


def recall_at_k(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    ground_truth_indices: list[int],
    k: int = 5,
) -> float:
    """Compute Recall@K.

    Args:
        query_embeddings: (N, D) normalized query vectors.
        target_embeddings: (M, D) normalized candidate vectors.
        ground_truth_indices: For each query, the index of the correct candidate.
        k: Number of top results to consider.

    Returns:
        Recall@K as a float in [0, 1].
    """
    scores = query_embeddings @ target_embeddings.T  # (N, M)
    top_k_indices = scores.topk(k, dim=1).indices  # (N, K)

    hits = 0
    for i, gt_idx in enumerate(ground_truth_indices):
        if gt_idx in top_k_indices[i]:
            hits += 1

    return hits / len(ground_truth_indices)


def mrr(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    ground_truth_indices: list[int],
    max_k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        query_embeddings: (N, D) normalized query vectors.
        target_embeddings: (M, D) normalized candidate vectors.
        ground_truth_indices: For each query, the index of the correct candidate.
        max_k: Maximum rank to consider.

    Returns:
        MRR as a float in [0, 1].
    """
    scores = query_embeddings @ target_embeddings.T  # (N, M)
    top_k_indices = scores.topk(max_k, dim=1).indices  # (N, max_k)

    reciprocal_ranks = []
    for i, gt_idx in enumerate(ground_truth_indices):
        matches = (top_k_indices[i] == gt_idx).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            rank = matches[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_retriever(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    ground_truth_indices: list[int],
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """Run full evaluation suite.

    Returns:
        Dictionary with Recall@K for each K and MRR.
    """
    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(
            query_embeddings, target_embeddings, ground_truth_indices, k
        )
    results["mrr"] = mrr(
        query_embeddings, target_embeddings, ground_truth_indices, max_k=max(k_values)
    )
    return results
