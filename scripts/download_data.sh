#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"

echo "=== Downloading SlideVQA ==="
# Clone the SlideVQA repository for annotations
if [ ! -d "$DATA_DIR/slidevqa" ]; then
    git clone https://github.com/nttmdlab-nlp/SlideVQA.git "$DATA_DIR/slidevqa"
else
    echo "SlideVQA already downloaded, skipping."
fi

echo "=== Downloading DocVQA ==="
# DocVQA is loaded via HuggingFace datasets (automatic download)
# This script just triggers the cache download
python3 -c "
from datasets import load_dataset
print('Downloading DocVQA validation split...')
ds = load_dataset('lmms-lab/DocVQA', 'DocVQA', split='validation', cache_dir='$DATA_DIR/hf_cache')
print(f'Downloaded {len(ds)} validation samples.')
ds = load_dataset('lmms-lab/DocVQA', 'DocVQA', split='train', cache_dir='$DATA_DIR/hf_cache')
print(f'Downloaded {len(ds)} training samples.')
"

echo "=== Done ==="
