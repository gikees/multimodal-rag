"""Process documents: PDF → page images → layout detection → region extraction."""

import argparse
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from src.document.layout import LayoutDetector, save_regions
from src.document.parser import pdf_to_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_pdf(
    pdf_path: Path,
    detector: LayoutDetector,
    output_dir: Path,
    dpi: int = 200,
) -> list[dict]:
    """Process a single PDF: parse → detect → extract regions."""
    doc_name = pdf_path.stem
    logger.info(f"Processing: {doc_name}")

    page_images = pdf_to_images(pdf_path, dpi=dpi)
    regions = detector.process_pages(page_images)
    region_paths = save_regions(regions, output_dir / doc_name, doc_name)

    metadata = []
    for region, path in zip(regions, region_paths):
        metadata.append({
            "doc_name": doc_name,
            "page_index": region.page_index,
            "region_index": region.region_index,
            "label": region.label,
            "bbox": list(region.bbox),
            "confidence": region.confidence,
            "image_path": str(path),
        })
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Process documents into regions")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-dir", required=True, help="Directory with PDF files")
    parser.add_argument("--output-dir", default=None, help="Output directory for regions")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    output_dir = Path(args.output_dir or cfg.paths.regions_dir)

    detector = LayoutDetector(
        model_path=cfg.document.layout_model,
        conf_threshold=cfg.document.layout_conf_threshold,
        padding=cfg.document.region_padding,
    )

    input_dir = Path(args.input_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    all_metadata = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        metadata = process_pdf(pdf_path, detector, output_dir, dpi=cfg.document.dpi)
        all_metadata.extend(metadata)

    # Save metadata index
    index_path = output_dir / "region_index.json"
    with open(index_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"Saved {len(all_metadata)} regions. Index: {index_path}")


if __name__ == "__main__":
    main()
