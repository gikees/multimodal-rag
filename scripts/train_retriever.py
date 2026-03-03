"""Train the custom cross-modal retriever."""

import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.retriever.baseline import SigLIPRetriever
from src.retriever.custom import CustomRetriever
from src.training.dataset import ContrastiveDataset, build_hard_negatives
from src.training.train import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train custom retriever")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-data", required=True, help="Training data JSON (queries + positive regions)")
    parser.add_argument("--val-data", default=None, help="Validation data JSON")
    parser.add_argument("--region-index", required=True, help="Path to region_index.json")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load training data
    with open(args.train_data) as f:
        train_data = json.load(f)
    queries = [d["question"] for d in train_data]
    positive_paths = [d["region_path"] for d in train_data]
    positive_indices = [d["region_index"] for d in train_data]

    # Load all region paths
    with open(args.region_index) as f:
        all_regions = json.load(f)
    all_region_paths = [r["image_path"] for r in all_regions]

    # Mine hard negatives with baseline retriever
    logger.info("Mining hard negatives with baseline retriever...")
    baseline = SigLIPRetriever(model_name=cfg.retriever.model_name, device=device)
    from PIL import Image
    all_images = [Image.open(p).convert("RGB") for p in all_region_paths]
    image_embeddings = baseline.embed_images(all_images)

    hard_neg_paths = build_hard_negatives(
        queries=queries,
        positive_indices=positive_indices,
        all_region_paths=all_region_paths,
        retriever=baseline,
        image_embeddings=image_embeddings,
        top_k=20,
        num_negatives=cfg.training.num_hard_negatives,
    )

    # Build dataset and dataloader
    train_dataset = ContrastiveDataset(
        queries=queries,
        positive_image_paths=positive_paths,
        hard_negative_paths=hard_neg_paths,
        processor_name=cfg.retriever.model_name,
        num_negatives=cfg.training.num_hard_negatives,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Build validation loader
    val_loader = None
    if args.val_data:
        with open(args.val_data) as f:
            val_data = json.load(f)
        val_queries = [d["question"] for d in val_data]
        val_pos_paths = [d["region_path"] for d in val_data]
        val_neg_paths = build_hard_negatives(
            queries=val_queries,
            positive_indices=[d["region_index"] for d in val_data],
            all_region_paths=all_region_paths,
            retriever=baseline,
            image_embeddings=image_embeddings,
            num_negatives=cfg.training.num_hard_negatives,
        )
        val_dataset = ContrastiveDataset(
            val_queries, val_pos_paths, val_neg_paths,
            processor_name=cfg.retriever.model_name,
            num_negatives=cfg.training.num_hard_negatives,
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, num_workers=4)

    # Initialize custom retriever
    model = CustomRetriever(
        model_name=cfg.retriever.model_name,
        projection_dim=cfg.training.projection_dim,
        freeze_layers=cfg.training.freeze_layers,
    )

    # Train
    logger.info("Starting training...")
    trained_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        checkpoint_dir=cfg.paths.checkpoints_dir,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        use_wandb=args.wandb,
        device=device,
    )
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
