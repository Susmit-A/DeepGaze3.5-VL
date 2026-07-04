# InternVL3.5-8B Combined Scanpath Model

📄 **Paper:** https://arxiv.org/abs/2607.02083

A rank-32 LoRA adapter on top of `OpenGVLab/InternVL3_5-8B-HF`, fine-tuned to
predict human free-viewing scanpaths as coordinate sequences on a 100x100 grid.
Given an image, the model emits a sequence of fixation coordinates that
approximate where a human observer would look. This is the **combined** model:
it was trained jointly on five eye-tracking datasets (MIT, CAT, COCO, Daemons,
Figrim). The weights bundled here are the best checkpoint, training step 13000.

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

## What is included

- `evaluate_vllm_unified.py` — the evaluation / scoring script.
- `configs/internvl3_5_8b_combined.yaml`, `configs/internvl3_5_8b_visual_search.yaml`
  — the LoRA SFT training configurations for the two models.
- `model/combined_adapter/` — the inference-only free-viewing LoRA adapter
  (rank 32; weights + tokenizer / processor config).
- `model/visual_search_adapter/` — the inference-only visual-search LoRA
  adapter (rank 8, COCO-Search18).
- `data/sample_MIT.json` — 75 validation entries (all subjects) for 5 MIT
  sample images.
- `data/images/` — the 5 sample images (`MIT_0985.jpg` .. `MIT_0989.jpg`).
- `data/centerbias/MIT/` — minimal per-image center-bias priors used as the
  baseline for the Information-Gain metric (only the `centerbias` array is
  retained; raw human-fixation and image arrays have been stripped).

## What is NOT included

- The full training/validation datasets (only a 5-image sample is bundled).
- The base-model weights. `OpenGVLab/InternVL3_5-8B-HF` (~16GB) is downloaded
  from HuggingFace on first run and the LoRA adapter is merged locally.

## Directory layout

```
internvl3_5_8b_combined_release/
├── README.md
├── LICENSE
├── requirements.txt
├── run_eval.sh
├── .gitignore
├── evaluate_vllm_unified.py
├── configs/
│   ├── internvl3_5_8b_combined.yaml
│   └── internvl3_5_8b_visual_search.yaml
├── model/
│   ├── combined_adapter/          # free-viewing scanpath (rank 32)
│   └── visual_search_adapter/     # COCO-Search18 visual search (rank 8)
├── data/
│   ├── sample_MIT.json            # 75 entries for the 5 sample images
│   ├── images/                    # MIT_0985.jpg .. MIT_0989.jpg
│   └── centerbias/
│       └── MIT/                   # 0985.pkl .. 0989.pkl (centerbias only)
└── eval_output/                   # created at runtime (git-ignored)
```

## Quick start

1. Create an environment from the pinned requirements, or simply reuse the
   validated `vllm` conda env:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the evaluation (needs 1 GPU). The first run downloads the base model
   from HuggingFace and merges the LoRA adapter into
   `model/combined_adapter_merged/` (git-ignored):

   ```bash
   bash run_eval.sh
   ```

   Results are written to `eval_output/`.

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
    --metric-mode assume_normalized --normalize-digits \
    --batch-size 64 --max-num-seqs 32 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --skip-viz
```

On the first run the base model is downloaded from HuggingFace and the LoRA
adapter is merged into `model/combined_adapter_merged/` (reused on later runs).
A GPU is required.

### Metric modes

Two scoring modes are available via `--metric-mode`:

- **`assume_normalized`** (default, fast) — scores only the ground-truth
  coordinate of each fixation by probing its digits, assuming the per-digit
  distributions are already normalised (`log Z = 0`). Use together with
  `--normalize-digits`. Reports per-fixation Information Gain (IG) and
  log-likelihood (LL). Recommended default.
- **`grid`** — builds the full 100×100 next-fixation probability grid for every
  transition and normalises over it. Slower (many more forward passes) but also
  yields AUC and NSS alongside IG/LL, and can dump the grids with `--save-grids`.

For this model the two modes yield essentially identical IG/LL.

### Key options

| Flag | Meaning |
|------|---------|
| `--metric-mode {assume_normalized,grid}` | Scoring mode (default `assume_normalized`). |
| `--normalize-digits` / `--no-normalize-digits` | Renormalise each digit distribution over 0–9 (on by default). |
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

### Output

Results are written to `<output-dir>/unified_<mode>_0shot_<timestamp>.json`. The
`results` list has one entry per evaluated sample, with the ground-truth
fixations and per-fixation scores:

- `lp_mean_ig`, `lp_fixation_igs` — mean and per-fixation Information Gain (bits).
- `lp_mean_ll`, `lp_fixation_lls` — mean and per-fixation log-likelihood.
- In `grid` mode, AUC and NSS are additionally reported per fixation.

Aggregate IG/LL across the set are printed to stdout at the end of the run.

## Metric note

The default evaluation runs with `--metric-mode assume_normalized` and
`--normalize-digits`. This computes a fast per-fixation Information Gain (IG) /
log-likelihood. IG is reported in **bits per fixation** relative to a per-image
center-bias baseline (the priors in `data/centerbias/`). Positive IG means the
model predicts fixations better than the center-bias prior alone.

## How scoring works

Predicted fixation coordinates are probed digit-by-digit from the model's
output distribution. With `--normalize-digits`, each digit's distribution is
renormalized over the 10 digit tokens (0-9) before the coordinate
log-likelihood is accumulated, so probability mass on non-digit tokens does not
distort the score.
