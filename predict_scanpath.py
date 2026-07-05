#!/usr/bin/env python3
"""
Scanpath inference for the InternVL3.5-8B combined model.

Given a single image, predict a human eye-movement scanpath. Supports two modes:
  - freeview: free-viewing scanpath (default 8 fixations)
  - search:   visual-search scanpath toward a target object (default 3 fixations)

This is a pure inference script: image -> generated scanpath. It computes no
metrics (no Information Gain / AUC / NSS), uses no centerbias, and does no grid
probing. A single autoregressive generation call produces the scanpath.

Model loading and prompt construction exactly mirror the evaluation path in
evaluate_vllm_unified.py (imported, not reimplemented).

Usage:
    python predict_scanpath.py --image path/to/image.jpg --mode freeview
    python predict_scanpath.py --image path/to/image.jpg --mode search --target toilet
"""

import argparse
import json
import math
import os
import re
import sys

from PIL import Image

from evaluate_vllm_unified import (
    load_model_vllm,
    FewShotPromptBuilder,
    parse_scanpath_reduced,
)


# =============================================================================
# Prompt templates (copied VERBATIM from the training data). The chat template
# adds the image, so these strings must NOT contain "<image>".
# =============================================================================

def build_freeview_prompt(n):
    """Return the free-viewing text prompt for exactly n fixations."""
    return (
        f"Analyze this image and predict a human eye movement scanpath during free viewing for 3 seconds.\n"
        f"A scanpath is the temporal sequence of fixation points showing where a person looks over time.\n"
        f"Consider visual saliency, semantic importance, and how attention naturally flows across a scene.\n"
        f"\n"
        f"Generate a scanpath of exactly {n} fixation points in temporal order as a list of tuples: (x, y)\n"
        f"- x: horizontal position (0-100, 0=left, 100=right)\n"
        f"- y: vertical position (0-100, 0=top, 100=bottom)\n"
        f"- Points should be ordered from first fixation to last fixation.\n"
        f"\n"
        f"Output ONLY a Python list of tuples:\n"
        f"[(51,46),(38,28),...]"
    )


def build_search_prompt(target, n):
    """Return the visual-search text prompt for a given target and n fixations."""
    return (
        f"Analyze this image and predict a human eye movement scanpath while searching for a {target}.\n"
        f"A scanpath is the temporal sequence of fixation points showing where a person looks over time.\n"
        f"Consider the search target, visual saliency, and how attention naturally flows during visual search.\n"
        f"\n"
        f"Generate a scanpath of exactly {n} fixation points in temporal order as a list of tuples: (x, y)\n"
        f"- x: horizontal position (0-100, 0=left, 100=right)\n"
        f"- y: vertical position (0-100, 0=top, 100=bottom)\n"
        f"- Points should be ordered from first fixation to last fixation.\n"
        f"\n"
        f"Output ONLY a Python list of tuples:\n"
        f"[(51,46),(38,28),...]"
    )


# The 18 COCO-Search18 targets the search adapter was trained on.
COCO_SEARCH18_TARGETS = [
    "bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard",
    "knife", "laptop", "microwave", "mouse", "oven", "potted plant", "sink",
    "stop sign", "toilet", "tv",
]

DEFAULT_BASE_MODEL = "OpenGVLab/InternVL3_5-8B-HF"


# =============================================================================
# Overlay rendering
# =============================================================================

def save_overlay(image, pixel_coords, out_path):
    """Draw the numbered fixations and connecting path over the image and save."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H = image.size
    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax.imshow(image)
    ax.axis("off")

    if pixel_coords:
        xs = [p[0] for p in pixel_coords]
        ys = [p[1] for p in pixel_coords]
        ax.plot(xs, ys, "-", color="cyan", linewidth=2, alpha=0.8, zorder=2)
        ax.scatter(xs, ys, s=300, facecolors="none", edgecolors="red",
                   linewidths=2, zorder=3)
        for i, (px, py) in enumerate(pixel_coords):
            ax.text(px, py, str(i + 1), color="yellow", fontsize=10,
                    ha="center", va="center", weight="bold", zorder=4)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a human eye-movement scanpath for a single image "
                    "(freeview or visual search)."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--mode", choices=["freeview", "search"], default="freeview",
        help="Prediction mode (default: freeview)."
    )
    parser.add_argument(
        "--target", default=None,
        help="Search target object (required iff --mode search); inserted as "
             "'a {target}'."
    )
    parser.add_argument(
        "--num-fixations", type=int, default=None,
        help="Number of fixations to generate. Default: 8 for freeview, 3 for search."
    )
    parser.add_argument(
        "--adapter-path", default=None,
        help="LoRA adapter path. Default: <script_dir>/model/combined_adapter "
             "(freeview) or <script_dir>/model/visual_search_adapter (search)."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help=f"Base model (default: {DEFAULT_BASE_MODEL}).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum context length (default: 4096).")
    parser.add_argument("--max-num-seqs", type=int, default=32,
                        help="Maximum concurrent sequences (default: 32).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (default: 0.90).")
    parser.add_argument("--output", default=None,
                        help="Optional path to write a JSON result file.")
    parser.add_argument("--save-overlay", default=None,
                        help="Optional path to save a PNG overlay of the scanpath.")

    args = parser.parse_args()

    if args.mode == "search" and not args.target:
        parser.error("--target is required when --mode is 'search'.")

    return args


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve num_fixations default based on mode.
    if args.num_fixations is None:
        n = 8 if args.mode == "freeview" else 3
    else:
        n = args.num_fixations

    # Resolve adapter-path default based on mode.
    if args.adapter_path is None:
        if args.mode == "freeview":
            adapter_path = os.path.join(script_dir, "model", "combined_adapter")
        else:
            adapter_path = os.path.join(script_dir, "model", "visual_search_adapter")
    else:
        adapter_path = args.adapter_path

    # Build the prompt text and warn about unknown search targets.
    if args.mode == "freeview":
        prompt_text = build_freeview_prompt(n)
    else:
        if args.target not in COCO_SEARCH18_TARGETS:
            print(
                f"Warning: search target '{args.target}' is not one of the 18 "
                f"trained COCO-Search18 targets ({', '.join(COCO_SEARCH18_TARGETS)}). "
                f"Proceeding anyway.",
                file=sys.stderr,
            )
        prompt_text = build_search_prompt(args.target, n)

    # Load the image.
    img = Image.open(args.image).convert("RGB")
    W, H = img.size

    # Load the model, mirroring the eval path exactly.
    llm, _ = load_model_vllm(
        base_model=args.base_model,
        adapter_path=adapter_path,
        num_images=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    builder = FewShotPromptBuilder(processor)

    prompt_str, mm_data = builder.build_prompt(
        test_image=img,
        test_prompt=prompt_text,
        shot_examples=[],
        partial_response="",
    )

    # Single autoregressive generation call.
    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=max(64, 16 * n + 16),
        seed=args.seed,
    )
    out = llm.generate([{"prompt": prompt_str, "multi_modal_data": mm_data}], sp)
    text = out[0].outputs[0].text

    coords = parse_scanpath_reduced(text)  # list of (x, y) ints on 0..99 grid

    # Scale grid coordinates (0-100 space) to pixel space.
    pixel_coords = [
        (int(round(x / 100.0 * W)), int(round(y / 100.0 * H)))
        for x, y in coords
    ]

    # Minimal stdout.
    print(f"Mode: {args.mode}")
    if args.mode == "search":
        print(f"Target: {args.target}")
    print(f"Fixations requested: {n}, generated: {len(coords)}")
    print(f"Image size (WxH): {W}x{H}")
    print(f"Scanpath (0-100 grid): {coords}")
    print(f"Scanpath (pixels):     {pixel_coords}")

    if args.output:
        result = {
            "mode": args.mode,
            "target": args.target,
            "num_fixations": n,
            "image": args.image,
            "scanpath_grid": coords,
            "scanpath_pixels": pixel_coords,
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote JSON: {args.output}")

    if args.save_overlay:
        save_overlay(img, pixel_coords, args.save_overlay)
        print(f"Wrote overlay: {args.save_overlay}")


if __name__ == "__main__":
    main()
