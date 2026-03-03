"""PDF/slide document parsing — converts pages to images using PyMuPDF."""

from pathlib import Path

import fitz
from PIL import Image


def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """Convert each page of a PDF to a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering. Higher = better quality but slower.

    Returns:
        List of PIL Images, one per page.
    """
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def save_page_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
) -> list[Path]:
    """Convert PDF pages to images and save them to disk.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save page images.
        dpi: Resolution for rendering.

    Returns:
        List of paths to saved images.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = pdf_to_images(pdf_path, dpi)
    saved = []
    for i, img in enumerate(images):
        out_path = output_dir / f"{pdf_path.stem}_page_{i:03d}.png"
        img.save(out_path)
        saved.append(out_path)
    return saved
