"""Training dataset: (query, positive_region, hard_negatives) triplets."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class ContrastiveDataset(Dataset):
    """Dataset of (query, positive_image, hard_negative_images) triplets.

    Each sample yields processor-ready inputs for the text query, the positive
    region image, and a set of hard negative region images.
    """

    def __init__(
        self,
        queries: list[str],
        positive_image_paths: list[str],
        hard_negative_paths: list[list[str]],
        processor_name: str = "google/siglip-base-patch16-224",
        num_negatives: int = 7,
    ):
        """
        Args:
            queries: Text queries.
            positive_image_paths: Path to the correct region image for each query.
            hard_negative_paths: For each query, a list of paths to hard negative images.
            processor_name: HuggingFace processor for tokenization and image preprocessing.
            num_negatives: Number of hard negatives to sample per query.
        """
        assert len(queries) == len(positive_image_paths) == len(hard_negative_paths)
        self.queries = queries
        self.positive_image_paths = positive_image_paths
        self.hard_negative_paths = hard_negative_paths
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        query = self.queries[idx]
        pos_image = Image.open(self.positive_image_paths[idx]).convert("RGB")

        # Sample hard negatives (pad with duplicates if not enough)
        neg_paths = self.hard_negative_paths[idx]
        if len(neg_paths) < self.num_negatives:
            neg_paths = neg_paths * ((self.num_negatives // max(len(neg_paths), 1)) + 1)
        neg_paths = neg_paths[: self.num_negatives]
        neg_images = [Image.open(p).convert("RGB") for p in neg_paths]

        # Process text
        text_inputs = self.processor(
            text=[query], return_tensors="pt", padding="max_length",
            truncation=True, max_length=64,
        )

        # Process images: positive + negatives
        all_images = [pos_image] + neg_images
        image_inputs = self.processor(images=all_images, return_tensors="pt")

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"],  # (1+N, C, H, W)
        }


def build_hard_negatives(
    queries: list[str],
    positive_indices: list[int],
    all_region_paths: list[str],
    retriever,
    image_embeddings: torch.Tensor,
    top_k: int = 20,
    num_negatives: int = 7,
) -> list[list[str]]:
    """Mine hard negatives using the baseline retriever.

    For each query, retrieve top-K candidates and keep those that are NOT the
    ground-truth positive.

    Args:
        queries: Text queries.
        positive_indices: Index of the correct region for each query.
        all_region_paths: Paths to all region images.
        retriever: Baseline retriever with .embed_texts() method.
        image_embeddings: Pre-computed (M, D) image embeddings.
        top_k: Number of candidates to retrieve per query.
        num_negatives: Number of hard negatives to keep per query.

    Returns:
        List of lists of hard negative image paths.
    """
    query_embs = retriever.embed_texts(queries)
    scores = query_embs @ image_embeddings.T  # (N, M)

    hard_negs = []
    for i, gt_idx in enumerate(positive_indices):
        top_indices = scores[i].argsort(descending=True).tolist()
        # Filter out the positive
        neg_indices = [j for j in top_indices if j != gt_idx][:num_negatives]
        hard_negs.append([all_region_paths[j] for j in neg_indices])

    return hard_negs
