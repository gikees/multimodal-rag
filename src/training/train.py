"""Contrastive fine-tuning loop for the custom cross-modal retriever."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.retriever.custom import CustomRetriever
from src.training.losses import info_nce_loss

logger = logging.getLogger(__name__)


def train(
    model: CustomRetriever,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    gradient_checkpointing: bool = True,
    checkpoint_dir: str = "checkpoints",
    eval_steps: int = 500,
    save_steps: int = 1000,
    temperature: float = 0.07,
    use_wandb: bool = False,
    device: str = "cuda",
) -> CustomRetriever:
    """Train the custom retriever with contrastive loss.

    Args:
        model: CustomRetriever model.
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        warmup_ratio: Fraction of steps for warmup.
        gradient_accumulation_steps: Accumulate gradients over this many steps.
        fp16: Use mixed precision training.
        gradient_checkpointing: Enable gradient checkpointing.
        checkpoint_dir: Directory to save checkpoints.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.
        temperature: InfoNCE temperature.
        use_wandb: Log to Weights & Biases.
        device: Device to train on.

    Returns:
        Trained model.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    if gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()

    # Only optimize unfrozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    scaler = GradScaler(enabled=fp16)

    if use_wandb:
        import wandb
        wandb.init(project="multimodal-rag-retriever")

    global_step = 0
    best_val_recall = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(device)  # (B, 1+N, C, H, W)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Split positive and negative images
            pos_pixels = pixel_values[:, 0]  # (B, C, H, W)
            neg_pixels = pixel_values[:, 1:]  # (B, N, C, H, W)
            B, N, C, H, W = neg_pixels.shape

            with autocast(enabled=fp16):
                # Encode queries and positive images
                text_embs = model.encode_texts(input_ids, attention_mask)
                pos_embs = model.encode_images(pos_pixels)

                # Encode negative images
                neg_flat = neg_pixels.reshape(B * N, C, H, W)
                neg_embs = model.encode_images(neg_flat)
                neg_embs = neg_embs.reshape(B, N, -1)

                loss = info_nce_loss(text_embs, pos_embs, neg_embs, temperature)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Linear warmup
                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = learning_rate * lr_scale
                else:
                    scheduler.step()

                global_step += 1

            epoch_loss += loss.item() * gradient_accumulation_steps
            progress.set_postfix(loss=f"{loss.item() * gradient_accumulation_steps:.4f}")

            if use_wandb:
                import wandb
                wandb.log({"train/loss": loss.item() * gradient_accumulation_steps, "step": global_step})

            # Evaluation
            if val_loader is not None and global_step > 0 and global_step % eval_steps == 0:
                val_recall = _evaluate(model, val_loader, device, fp16)
                logger.info(f"Step {global_step} - Val Recall@5: {val_recall:.4f}")
                if use_wandb:
                    import wandb
                    wandb.log({"val/recall@5": val_recall, "step": global_step})
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
                model.train()

            # Save checkpoint
            if global_step > 0 and global_step % save_steps == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                    },
                    checkpoint_dir / f"checkpoint_{global_step}.pt",
                )

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")
    return model


@torch.no_grad()
def _evaluate(
    model: CustomRetriever,
    val_loader: DataLoader,
    device: str,
    fp16: bool,
    top_k: int = 5,
) -> float:
    """Quick evaluation: compute Recall@K on validation set."""
    model.eval()
    hits = 0
    total = 0

    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pos_pixels = pixel_values[:, 0]
        all_pixels = pixel_values.reshape(-1, *pixel_values.shape[2:])

        with autocast(enabled=fp16):
            text_embs = model.encode_texts(input_ids, attention_mask)
            all_embs = model.encode_images(all_pixels)

        B = text_embs.size(0)
        num_per_query = pixel_values.size(1)
        all_embs = all_embs.reshape(B, num_per_query, -1)

        for i in range(B):
            scores = text_embs[i] @ all_embs[i].T
            top_indices = scores.argsort(descending=True)[:top_k]
            if 0 in top_indices:  # Index 0 is the positive
                hits += 1
            total += 1

    return hits / max(total, 1)
