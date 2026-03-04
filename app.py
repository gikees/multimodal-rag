"""Gradio web UI for the Multimodal RAG system."""

import json
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from src.document.layout import LayoutDetector, Region
from src.document.parser import pdf_to_images
from src.retriever.baseline import SigLIPRetriever
from src.vectordb.store import RegionMetadata, VectorStore

# Load config
cfg = OmegaConf.load("configs/default.yaml")

# Global state
retriever = None
store = None
detector = None
generator = None
indexed_regions: list[dict] = []
page_images: list[Image.Image] = []


def initialize():
    """Initialize models on startup."""
    global retriever, store, detector

    retriever = SigLIPRetriever(model_name=cfg.retriever.model_name)
    detector = LayoutDetector(
        model_path=cfg.document.layout_model,
        conf_threshold=cfg.document.layout_conf_threshold,
    )

    try:
        store = VectorStore(
            host=cfg.vectordb.host,
            port=cfg.vectordb.port,
            collection_name=cfg.vectordb.collection_name,
        )
        store.create_collection(cfg.retriever.embedding_dim)
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant: {e}")
        print("Running without vector store — using in-memory search.")
        store = None


# In-memory fallback when Qdrant isn't running
region_embeddings: torch.Tensor | None = None
region_metadata: list[dict] = []


def process_document(file) -> str:
    """Process an uploaded PDF: parse → detect → index."""
    global page_images, indexed_regions, region_embeddings, region_metadata

    if file is None:
        return "No file uploaded."

    pdf_path = Path(file.name)
    doc_name = pdf_path.stem

    # Parse PDF to images
    page_images = pdf_to_images(pdf_path, dpi=cfg.document.dpi)
    status = f"Parsed {len(page_images)} pages.\n"

    # Detect regions
    all_regions = detector.process_pages(page_images)
    status += f"Detected {len(all_regions)} regions.\n"

    if not all_regions:
        return status + "No regions detected."

    # Embed regions
    region_images = [r.image for r in all_regions]
    embeddings = retriever.embed_images(region_images)

    # Store metadata
    region_metadata.clear()
    for r in all_regions:
        region_metadata.append({
            "doc_name": doc_name,
            "page_index": r.page_index,
            "region_index": r.region_index,
            "label": r.label,
            "bbox": [int(x) for x in r.bbox],
            "confidence": r.confidence,
        })

    if store is not None:
        emb_np = embeddings.numpy()
        meta_list = [
            RegionMetadata(
                doc_name=doc_name,
                page_index=r.page_index,
                region_index=r.region_index,
                label=r.label,
                bbox=list(r.bbox),
                image_path="",
            )
            for r in all_regions
        ]
        store.upsert_regions(emb_np, meta_list)
        status += f"Indexed {len(all_regions)} regions into Qdrant.\n"
    else:
        region_embeddings = embeddings
        status += f"Indexed {len(all_regions)} regions (in-memory).\n"

    indexed_regions.clear()
    indexed_regions.extend([
        {"region": r, "embedding": embeddings[i]} for i, r in enumerate(all_regions)
    ])

    return status + "Ready for queries!"


def query_document(question: str, top_k: int = 3) -> tuple[str, Image.Image | None]:
    """Query the indexed document and return answer + highlighted region."""
    if not indexed_regions:
        return "No document indexed. Please upload a PDF first.", None

    # Embed query
    query_emb = retriever.embed_texts([question])

    # Search
    if store is not None:
        results = store.search(query_emb[0].numpy(), top_k=top_k)
    else:
        scores = (query_emb @ region_embeddings.T).squeeze(0)
        top_indices = scores.argsort(descending=True)[:top_k]
        results = [
            {**region_metadata[idx], "score": scores[idx].item()}
            for idx in top_indices
        ]

    if not results:
        return "No matching regions found.", None

    # Get the top result
    top = results[0]
    page_idx = top["page_index"]
    bbox = top["bbox"]

    # Highlight region on page
    page_img = page_images[page_idx].copy()
    draw = ImageDraw.Draw(page_img)
    draw.rectangle(bbox, outline="red", width=3)

    # Build response
    region_img = indexed_regions[0]["region"].image
    for item in indexed_regions:
        r = item["region"]
        if r.page_index == page_idx and list(r.bbox) == bbox:
            region_img = r.image
            break

    # Try to generate answer with Gemini
    answer = f"Top region: {top.get('label', 'unknown')} (score: {top.get('score', 0):.3f})\n"
    answer += f"Page {page_idx + 1}, bbox: {bbox}\n\n"

    try:
        if generator is None:
            _init_generator()
        if generator is not None:
            generated = generator.generate(question, region_img)
            answer += f"Answer: {generated}"
        else:
            answer += "(Generator not available)"
    except Exception as e:
        answer += f"(Generation failed: {e})"

    return answer, page_img


def _init_generator():
    """Lazily initialize the generator."""
    global generator
    try:
        from src.generator.local_vl import LocalVLGenerator
        generator = LocalVLGenerator(model_name=cfg.generator.model)
    except Exception as e:
        print(f"Warning: Could not load local VL generator: {e}")
        generator = None


def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="Multimodal RAG") as app:
        gr.Markdown("# Multimodal RAG System\nUpload a PDF and ask questions about its visual content.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                process_btn = gr.Button("Process Document", variant="primary")
                status_output = gr.Textbox(label="Status", lines=4)

            with gr.Column(scale=1):
                question_input = gr.Textbox(label="Question", placeholder="Ask about the document...")
                top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top-K regions")
                query_btn = gr.Button("Ask", variant="primary")

        with gr.Row():
            answer_output = gr.Textbox(label="Answer", lines=6)
            image_output = gr.Image(label="Retrieved Region (highlighted)")

        process_btn.click(fn=process_document, inputs=[file_input], outputs=[status_output])
        query_btn.click(
            fn=query_document,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, image_output],
        )

    return app


if __name__ == "__main__":
    initialize()
    app = build_app()
    app.launch(server_name="0.0.0.0")
