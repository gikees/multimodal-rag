"""Embed document regions and index them into Qdrant."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from src.retriever.baseline import SigLIPRetriever
from src.vectordb.store import RegionMetadata, VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Index regions into Qdrant")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--region-index", required=True, help="Path to region_index.json")
    parser.add_argument("--use-custom", action="store_true", help="Use custom retriever")
    parser.add_argument("--checkpoint", default=None, help="Custom retriever checkpoint")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Load region metadata
    with open(args.region_index) as f:
        region_meta = json.load(f)
    logger.info(f"Loaded {len(region_meta)} regions")

    # Initialize retriever
    if args.use_custom and args.checkpoint:
        from src.retriever.custom import CustomRetriever
        import torch
        retriever = CustomRetriever(
            model_name=cfg.retriever.model_name,
            projection_dim=cfg.training.projection_dim,
            freeze_layers=cfg.training.freeze_layers,
        )
        retriever.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        retriever.eval()
        embedding_dim = cfg.training.projection_dim
    else:
        retriever = SigLIPRetriever(model_name=cfg.retriever.model_name)
        embedding_dim = cfg.retriever.embedding_dim

    # Initialize vector store
    store = VectorStore(
        host=cfg.vectordb.host,
        port=cfg.vectordb.port,
        collection_name=cfg.vectordb.collection_name,
    )
    store.create_collection(embedding_dim)

    # Embed and index in batches
    batch_size = 32
    for i in tqdm(range(0, len(region_meta), batch_size), desc="Indexing"):
        batch_meta = region_meta[i : i + batch_size]
        images = [Image.open(m["image_path"]).convert("RGB") for m in batch_meta]

        if args.use_custom:
            embeddings = retriever.embed_images(images, batch_size=batch_size)
            embeddings = embeddings.numpy()
        else:
            embeddings = retriever.embed_images(images, batch_size=batch_size).numpy()

        metadata_list = [
            RegionMetadata(
                doc_name=m["doc_name"],
                page_index=m["page_index"],
                region_index=m["region_index"],
                label=m["label"],
                bbox=m["bbox"],
                image_path=m["image_path"],
            )
            for m in batch_meta
        ]
        store.upsert_regions(embeddings, metadata_list, start_id=i)

    logger.info(f"Indexed {store.count()} regions into Qdrant")


if __name__ == "__main__":
    main()
