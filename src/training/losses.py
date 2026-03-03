"""Contrastive losses for cross-modal retriever training."""

import torch
import torch.nn.functional as F


def info_nce_loss(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE loss for contrastive learning.

    When negative_embeddings is None, uses in-batch negatives (other positives
    in the batch serve as negatives for each query).

    Args:
        query_embeddings: (B, D) normalized text query embeddings.
        positive_embeddings: (B, D) normalized positive image embeddings.
        negative_embeddings: Optional (B, N, D) hard negative image embeddings.
        temperature: Temperature scaling factor.

    Returns:
        Scalar loss.
    """
    batch_size = query_embeddings.size(0)

    if negative_embeddings is None:
        # In-batch negatives: similarity matrix (B, B)
        logits = (query_embeddings @ positive_embeddings.T) / temperature
        labels = torch.arange(batch_size, device=logits.device)
        return F.cross_entropy(logits, labels)

    # Explicit hard negatives
    # pos_sim: (B, 1)
    pos_sim = (query_embeddings * positive_embeddings).sum(dim=-1, keepdim=True) / temperature

    # neg_sim: (B, N)
    neg_sim = torch.bmm(
        negative_embeddings, query_embeddings.unsqueeze(-1)
    ).squeeze(-1) / temperature

    # Concat: (B, 1+N), positive is at index 0
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
