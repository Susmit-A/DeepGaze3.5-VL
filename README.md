# InternVL3.5-8B Combined Scanpath Model

A rank-32 LoRA adapter on top of `OpenGVLab/InternVL3_5-8B-HF`, fine-tuned to
predict human free-viewing scanpaths as coordinate sequences on a 100x100 grid.
Given an image, the model emits a sequence of fixation coordinates that
approximate where a human observer would look. This is the **combined** model:
it was trained jointly on five eye-tracking datasets (MIT, CAT, COCO, Daemons,
Figrim). The weights bundled here are the best checkpoint, training step 13000.

## What is included

- `evaluate_vllm_unified.py` — the evaluation / scoring script.
- `configs/internvl3_5_8b_combined.yaml` — the LoRA SFT training configuration
  (rank 32, alpha 64, 5-dataset joint training).
- `model/combined_adapter/` — the inference-only LoRA adapter (weights +
  tokenizer / processor config). Training-only files (optimizer, scheduler, RNG
  state, trainer state) are intentionally excluded.
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
│   └── internvl3_5_8b_combined.yaml
├── model/
│   └── combined_adapter/          # inference-only LoRA adapter
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

## Reference numbers

Dataset-level IG on the full 2%-subset per dataset, checkpoint 13000, with
`assume_normalized` + `normalize-digits` (bits/fixation):

| Dataset | IG (bits/fixation) |
|---------|--------------------|
| MIT     | 2.174              |
| CAT     | 2.02               |
| COCO    | 2.17               |
| Daemons | ~2.9               |
| Figrim  | 1.92               |

Note: the bundled 5-image sample is far smaller than these full per-dataset
subsets, so the IG you obtain from `run_eval.sh` on the sample will differ from
these dataset-level numbers.
