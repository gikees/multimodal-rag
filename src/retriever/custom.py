"""Custom cross-modal retriever with contrastive fine-tuned projection head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int = 768, projection_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CustomRetriever(nn.Module):
    """SigLIP backbone with trainable projection head for cross-modal retrieval.

    The backbone's early layers are frozen; only the last few transformer
    blocks and the projection head are trained with contrastive loss.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        projection_dim: int = 256,
        freeze_layers: int = 8,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        embedding_dim = self.backbone.config.vision_config.hidden_size
        self.image_projection = ProjectionHead(embedding_dim, projection_dim)
        self.text_projection = ProjectionHead(
            self.backbone.config.text_config.hidden_size, projection_dim
        )

        self._freeze_early_layers(freeze_layers)

    def _freeze_early_layers(self, num_layers: int) -> None:
        """Freeze the first `num_layers` of both vision and text encoders."""
        # Freeze vision encoder embeddings and early layers
        for param in self.backbone.vision_model.embeddings.parameters():
            param.requires_grad = False
        for layer in self.backbone.vision_model.encoder.layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze text encoder embeddings and early layers
        for param in self.backbone.text_model.embeddings.parameters():
            param.requires_grad = False
        for layer in self.backbone.text_model.encoder.layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through backbone + projection head.

        Returns L2-normalized embeddings of shape (B, projection_dim).
        """
        features = self.backbone.get_image_features(pixel_values=pixel_values)
        projected = self.image_projection(features)
        return F.normalize(projected, p=2, dim=-1)

    def encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode texts through backbone + projection head.

        Returns L2-normalized embeddings of shape (B, projection_dim).
        """
        features = self.backbone.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        projected = self.text_projection(features)
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning normalized image and text embeddings."""
        image_embs = self.encode_images(pixel_values)
        text_embs = self.encode_texts(input_ids, attention_mask)
        return image_embs, text_embs

    @torch.no_grad()
    def embed_images(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Convenience method: embed PIL images for inference."""
        self.eval()
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(device)
            embs = self.encode_images(pixel_values)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Convenience method: embed text queries for inference."""
        self.eval()
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            embs = self.encode_texts(input_ids, attention_mask)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)
