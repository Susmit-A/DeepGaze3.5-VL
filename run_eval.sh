#!/usr/bin/env bash
# Evaluate the InternVL3.5-8B combined scanpath model on the 5 bundled MIT sample images.
# Requires 1 GPU. On first run the base model (~16GB) is downloaded from HuggingFace and
# the LoRA adapter is merged into model/combined_adapter_merged/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")"

python evaluate_vllm_unified.py \
    --base-model OpenGVLab/InternVL3_5-8B-HF \
    --adapter-path model/combined_adapter \
    --val-json data/sample_MIT.json \
    --images-dir data \
    --pkl-dir data/centerbias \
    --output-dir eval_output \
    --metric-mode assume_normalized \
    --normalize-digits \
    --batch-size 64 \
    --max-num-seqs 32 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --skip-viz
