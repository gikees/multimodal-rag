# Multimodal RAG with Custom Cross-Modal Retriever

A multimodal Retrieval-Augmented Generation system for visual document question answering. Upload PDFs or slide decks, ask natural language questions, and get grounded answers backed by retrieved visual regions.

The centerpiece is a **custom cross-modal retriever** trained with contrastive learning (InfoNCE) that outperforms off-the-shelf SigLIP embeddings on document retrieval tasks.

## Architecture

```
PDF/Slides → Page Images → Layout Detection (YOLOv8) → Region Extraction
                                                              ↓
Query → Text Encoder ──→ Cross-Modal Similarity Search ← Region Embeddings
                              ↓
                    Top-K Regions → MLLM (Gemini Flash) → Grounded Answer
```

## Results

| Retriever | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| SigLIP (frozen) | — | — | — |
| SigLIP (fine-tuned) | — | — | — |

*Results will be filled in after training.*

## Quick Start

### Setup

```bash
# Clone
git clone https://github.com/<your-username>/multimodal-rag.git
cd multimodal-rag

# Install
pip install -e ".[dev]"

# Start Qdrant (optional, falls back to in-memory)
docker run -p 6333:6333 qdrant/qdrant
```

### Run the Demo

```bash
# Set API key for answer generation (optional)
export GOOGLE_API_KEY=your_key_here

# Launch
python app.py
```

### Process Documents

```bash
# Download datasets
bash scripts/download_data.sh

# Process PDFs into regions
python scripts/process_documents.py --input-dir data/raw/pdfs

# Index regions
python scripts/index_regions.py --region-index data/processed/regions/region_index.json
```

### Train Custom Retriever

```bash
python scripts/train_retriever.py \
    --train-data data/processed/train.json \
    --val-data data/processed/val.json \
    --region-index data/processed/regions/region_index.json \
    --wandb
```

### Evaluate

```bash
python scripts/evaluate.py \
    --eval-data data/processed/val.json \
    --region-index data/processed/regions/region_index.json \
    --checkpoint checkpoints/best_model.pt
```

## Tech Stack

- **Layout Detection**: YOLOv8 pretrained on DocLayNet
- **Embeddings**: SigLIP (google/siglip-base-patch16-224)
- **Training**: PyTorch + HuggingFace Transformers + InfoNCE contrastive loss
- **Vector DB**: Qdrant
- **Generator**: Google Gemini Flash
- **UI**: Gradio
- **Datasets**: SlideVQA, DocVQA
