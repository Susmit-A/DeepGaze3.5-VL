# InternVL3.5-8B Combined Scanpath Model

📄 **Paper:** https://arxiv.org/abs/2607.02083

A rank-32 LoRA adapter on top of `OpenGVLab/InternVL3_5-8B-HF`, fine-tuned to
predict human free-viewing scanpaths as coordinate sequences on a 100x100 grid.
Given an image, the model emits a sequence of fixation coordinates that
approximate where a human observer would look. This is the **combined** model:
it was trained jointly on five eye-tracking datasets (MIT, CAT, COCO, Daemons,
Figrim). The weights bundled here are the best checkpoint.

## Models

Two LoRA adapters are bundled, both on top of `OpenGVLab/InternVL3_5-8B-HF`:

- **`model/combined_adapter/`** — free-viewing scanpath prediction (rank 32),
  trained jointly on MIT, CAT, COCO, Daemons and Figrim. Used by `run_eval.sh`.
- **`model/visual_search_adapter/`** — goal-directed **visual search** (rank 8),
  trained on COCO-Search18 (target-present and target-absent trials); given a
  search target it predicts the search scanpath.

Choose which one to load with `--adapter-path`. The bundled 5-image sample and
`run_eval.sh` target the free-viewing model; the visual-search model expects
COCO-Search18-style inputs (a target category in the prompt), which are not
bundled here.

## Directory layout

```
internvl3_5_8b_combined_release/
├── README.md
├── LICENSE
├── requirements.txt
├── run_eval.sh
├── predict_scanpath.py                    # image -> scanpath inference
├── evaluate_vllm_unified.py               # evaluation / scoring script
├── configs/
│   ├── internvl3_5_8b_combined.yaml        # free-viewing LoRA SFT config
│   └── internvl3_5_8b_visual_search.yaml   # visual-search LoRA SFT config
├── model/
│   ├── combined_adapter/          # free-viewing scanpath (rank 32)
│   └── visual_search_adapter/     # COCO-Search18 visual search (rank 8)
└── data/
    ├── sample_MIT.json            # 75 entries for the 5 sample MIT images
    ├── images/                    # MIT_0985.jpg .. MIT_0989.jpg
    └── centerbias/
        └── MIT/                   # per-image center-bias priors (IG baseline)
```

## Quick start

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the evaluation (needs 1 GPU). The first run downloads the base model
   from HuggingFace and merges the LoRA adapter into
   `model/combined_adapter_merged/`:

   ```bash
   bash run_eval.sh
   ```

   Results are written to `eval_output/`.

## Predict a scanpath

`predict_scanpath.py` runs a single image through the model and prints the
predicted scanpath — no ground truth or metrics needed.

Free viewing (8 fixations by default, uses `model/combined_adapter`):

```bash
python predict_scanpath.py --image path/to/image.jpg
```

Visual search (3 fixations by default, uses `model/visual_search_adapter`) —
pass the target object:

```bash
python predict_scanpath.py --image path/to/image.jpg --mode search --target toilet
```

The scanpath is printed both on the model's 0–100 grid and in pixel
coordinates. Use `--num-fixations N` to change the length, `--output out.json`
to save the result, and `--save-overlay overlay.png` to render the scanpath on
the image. The visual-search model was trained on the 18 COCO-Search18 targets:
bottle, bowl, car, chair, clock, cup, fork, keyboard, knife, laptop, microwave,
mouse, oven, potted plant, sink, stop sign, toilet, tv.

## Running evaluations

`run_eval.sh` is a thin wrapper around `evaluate_vllm_unified.py`. To customise
a run, call the script directly:

```bash
python evaluate_vllm_unified.py \
    --base-model OpenGVLab/InternVL3_5-8B-HF \
    --adapter-path model/combined_adapter \
    --val-json data/sample_MIT.json \
    --images-dir data \
    --pkl-dir data/centerbias \
    --output-dir eval_output \
    --metric-mode fast \
    --batch-size 64 --max-num-seqs 32 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --skip-viz
```

On the first run the base model is downloaded from HuggingFace and the LoRA
adapter is merged into `model/combined_adapter_merged/` (reused on later runs).
A GPU is required.

### Metric modes

Two scoring modes are available via `--metric-mode`:

- **`fast`** (default) — scores the ground-truth coordinate of each fixation by
  probing its digits with per-digit normalisation. Reports per-fixation
  Information Gain (IG) and log-likelihood (LL). Recommended.
- **`grid`** — builds the full 100×100 next-fixation probability grid for every
  transition and normalises over it. Slower (many more forward passes) but also
  yields AUC and NSS alongside IG/LL, and can dump the grids with `--save-grids`.

Both modes probe the fixation coordinates digit-by-digit; the digit distribution
is renormalised over the ten digit tokens (0–9) so that probability mass on
non-digit tokens does not distort the score.

### Key options

| Flag | Meaning |
|------|---------|
| `--metric-mode {fast,grid}` | Scoring mode (default `fast`). |
| `--base-model` | HuggingFace base model (`OpenGVLab/InternVL3_5-8B-HF`). |
| `--adapter-path` | LoRA adapter directory (merged into `*_merged/` on first use). |
| `--val-json` | Evaluation set in LlamaFactory format (see below). |
| `--images-dir` | Base directory that image paths in the JSON resolve against. |
| `--pkl-dir` | Center-bias priors (the IG baseline); omit to fall back to a synthetic Gaussian center bias. |
| `--output-dir` | Where the results JSON is written. |
| `--max-samples N` | Evaluate only the first N entries (quick checks). |
| `--batch-size` / `--max-num-seqs` / `--max-model-len` | vLLM throughput / context knobs. |
| `--gpu-memory-utilization` | vLLM GPU memory fraction. |
| `--skip-viz` | Skip per-sample visualisation output. |
| `--seed` | RNG seed. |

Run `python evaluate_vllm_unified.py --help` for the complete list.

### Evaluating on your own data

Pass a `--val-json` in LlamaFactory format — a list of entries, each with an
image and a human/assistant turn pair:

```json
[
  {
    "images": ["images/MIT_0987.jpg"],
    "conversations": [
      {"from": "human", "value": "<image>Analyze this image and predict a human eye movement scanpath ..."},
      {"from": "gpt",   "value": "[(53, 48), (72, 32), (38, 54)]"}
    ]
  }
]
```

- Coordinates are integers on a **0–99 grid** (the image is treated as 100×100),
  written as `(x, y)` with `x` = column, `y` = row. The `gpt` turn holds the
  ground-truth scanpath the model is scored against.
- Image paths resolve relative to `--images-dir` (so `images/MIT_0987.jpg` with
  `--images-dir data` reads `data/images/MIT_0987.jpg`).
- Center-bias priors are looked up at `<pkl-dir>/<DATASET>/<index>.pkl` derived
  from the image name `DATASET_index.jpg` (e.g. `MIT_0987.jpg` →
  `data/centerbias/MIT/0987.pkl`). Each pickle is a dict
  `{"centerbias": <2-D log-density array>}`. If a prior is missing, a synthetic
  Gaussian center bias is used instead.
