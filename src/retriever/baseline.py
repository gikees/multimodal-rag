"""Baseline retriever using off-the-shelf SigLIP embeddings."""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, SiglipProcessor


class SigLIPRetriever:
    """Embed text queries and images using frozen SigLIP."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_images(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Embed a list of images into normalized vectors.

        Args:
            images: List of PIL Images.
            batch_size: Batch size for processing.

        Returns:
            Tensor of shape (N, embedding_dim), L2-normalized.
        """
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            all_embeddings.append(outputs.cpu())
        embeddings = torch.cat(all_embeddings, dim=0)
        return F.normalize(embeddings, p=2, dim=-1)

    @torch.no_grad()
    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Embed a list of text queries into normalized vectors.

        Args:
            texts: List of query strings.
            batch_size: Batch size for processing.

        Returns:
            Tensor of shape (N, embedding_dim), L2-normalized.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)
            all_embeddings.append(outputs.cpu())
        embeddings = torch.cat(all_embeddings, dim=0)
        return F.normalize(embeddings, p=2, dim=-1)

    def retrieve(
        self,
        query: str,
        image_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Retrieve top-K most similar images for a text query.

        Args:
            query: Text query string.
            image_embeddings: Pre-computed image embeddings (N, D).
            top_k: Number of results to return.

        Returns:
            List of (index, score) tuples, sorted by descending similarity.
        """
        query_emb = self.embed_texts([query])  # (1, D)
        scores = (query_emb @ image_embeddings.T).squeeze(0)  # (N,)
        top_indices = scores.argsort(descending=True)[:top_k]
        return [(idx.item(), scores[idx].item()) for idx in top_indices]
