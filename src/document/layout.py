"""Layout detection using YOLOv8 pretrained on DocLayNet.

Detects document regions (tables, figures, text blocks, charts) and
extracts cropped region images.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

DOCLAYNET_LABELS = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]


@dataclass
class Region:
    """A detected region within a document page."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    label: str
    confidence: float
    image: Image.Image
    page_index: int
    region_index: int

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class LayoutDetector:
    """Detect document layout regions using YOLOv8."""

    WEIGHT_FILENAME = "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"

    def __init__(
        self,
        model_path: str = "DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet",
        conf_threshold: float = 0.25,
        padding: int = 5,
    ):
        # If model_path looks like a HuggingFace repo ID, download weights first
        if "/" in model_path and not Path(model_path).exists():
            model_path = hf_hub_download(
                repo_id=model_path,
                filename=self.WEIGHT_FILENAME,
            )
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.padding = padding

    def detect(
        self,
        page_image: Image.Image,
        page_index: int = 0,
    ) -> list[Region]:
        """Detect layout regions in a page image.

        Args:
            page_image: PIL Image of a document page.
            page_index: Index of the page (for tracking).

        Returns:
            List of detected Region objects with cropped images.
        """
        results = self.model.predict(
            page_image,
            conf=self.conf_threshold,
            verbose=False,
        )
        regions = []
        if not results or results[0].boxes is None:
            return regions

        boxes = results[0].boxes
        w, h = page_image.size

        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())
            label = DOCLAYNET_LABELS[cls_id] if cls_id < len(DOCLAYNET_LABELS) else "Unknown"

            # Apply padding and clamp to image bounds
            x1 = max(0, xyxy[0] - self.padding)
            y1 = max(0, xyxy[1] - self.padding)
            x2 = min(w, xyxy[2] + self.padding)
            y2 = min(h, xyxy[3] + self.padding)

            cropped = page_image.crop((x1, y1, x2, y2))
            regions.append(Region(
                bbox=(x1, y1, x2, y2),
                label=label,
                confidence=conf,
                image=cropped,
                page_index=page_index,
                region_index=i,
            ))

        return regions

    def process_pages(self, page_images: list[Image.Image]) -> list[Region]:
        """Process multiple page images and return all detected regions."""
        all_regions = []
        for page_idx, page_img in enumerate(page_images):
            regions = self.detect(page_img, page_index=page_idx)
            all_regions.extend(regions)
        return all_regions


def save_regions(
    regions: list[Region],
    output_dir: str | Path,
    doc_name: str,
) -> list[Path]:
    """Save cropped region images to disk.

    Args:
        regions: List of Region objects.
        output_dir: Directory to save region images.
        doc_name: Document name for file naming.

    Returns:
        List of paths to saved region images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for region in regions:
        filename = (
            f"{doc_name}_p{region.page_index:03d}"
            f"_r{region.region_index:03d}_{region.label}.png"
        )
        path = output_dir / filename
        region.image.save(path)
        paths.append(path)
    return paths
