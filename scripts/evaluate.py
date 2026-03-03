"""Run evaluation suite: compare baseline vs custom retriever."""

import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from src.retriever.baseline import SigLIPRetriever
from src.retriever.evaluate import evaluate_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_data(data_path: str) -> tuple[list[str], list[int]]:
    """Load evaluation data: queries and ground-truth region indices."""
    with open(data_path) as f:
        data = json.load(f)
    queries = [d["question"] for d in data]
    gt_indices = [d["region_index"] for d in data]
    return queries, gt_indices


def evaluate_baseline(cfg, queries, gt_indices, all_region_paths, device):
    """Evaluate the baseline SigLIP retriever."""
    logger.info("Evaluating baseline (frozen SigLIP)...")
    retriever = SigLIPRetriever(model_name=cfg.retriever.model_name, device=device)

    images = [Image.open(p).convert("RGB") for p in tqdm(all_region_paths, desc="Loading images")]
    image_embs = retriever.embed_images(images)
    query_embs = retriever.embed_texts(queries)

    results = evaluate_retriever(query_embs, image_embs, gt_indices, k_values=cfg.eval.top_k)
    return results


def evaluate_custom(cfg, queries, gt_indices, all_region_paths, checkpoint_path, device):
    """Evaluate the custom fine-tuned retriever."""
    logger.info("Evaluating custom retriever...")
    from src.retriever.custom import CustomRetriever

    model = CustomRetriever(
        model_name=cfg.retriever.model_name,
        projection_dim=cfg.training.projection_dim,
        freeze_layers=cfg.training.freeze_layers,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    images = [Image.open(p).convert("RGB") for p in tqdm(all_region_paths, desc="Loading images")]
    image_embs = model.embed_images(images, device=device)
    query_embs = model.embed_texts(queries, device=device)

    results = evaluate_retriever(query_embs, image_embs, gt_indices, k_values=cfg.eval.top_k)
    return results


def print_comparison(baseline_results, custom_results):
    """Print a comparison table."""
    header = f"{'Retriever':<25} " + " ".join(f"{'Recall@' + str(k):<12}" for k in [1, 5, 10]) + f"{'MRR':<12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    row = f"{'SigLIP (frozen)':<25} "
    for k in [1, 5, 10]:
        key = f"recall@{k}"
        row += f"{baseline_results.get(key, 0):<12.4f} "
    row += f"{baseline_results.get('mrr', 0):<12.4f}"
    print(row)

    if custom_results:
        row = f"{'SigLIP (fine-tuned)':<25} "
        for k in [1, 5, 10]:
            key = f"recall@{k}"
            row += f"{custom_results.get(key, 0):<12.4f} "
        row += f"{custom_results.get('mrr', 0):<12.4f}"
        print(row)

    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrievers")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--eval-data", required=True, help="Evaluation data JSON")
    parser.add_argument("--region-index", required=True, help="Path to region_index.json")
    parser.add_argument("--checkpoint", default=None, help="Custom retriever checkpoint")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    queries, gt_indices = load_eval_data(args.eval_data)

    with open(args.region_index) as f:
        all_regions = json.load(f)
    all_region_paths = [r["image_path"] for r in all_regions]

    # Limit eval samples
    max_samples = cfg.eval.num_val_samples
    if max_samples and len(queries) > max_samples:
        queries = queries[:max_samples]
        gt_indices = gt_indices[:max_samples]

    baseline_results = evaluate_baseline(cfg, queries, gt_indices, all_region_paths, device)

    custom_results = None
    if args.checkpoint:
        custom_results = evaluate_custom(
            cfg, queries, gt_indices, all_region_paths, args.checkpoint, device
        )

    print_comparison(baseline_results, custom_results)

    # Save results
    output = {
        "baseline": baseline_results,
        "custom": custom_results,
        "num_queries": len(queries),
        "num_regions": len(all_region_paths),
    }
    output_path = Path("eval_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
