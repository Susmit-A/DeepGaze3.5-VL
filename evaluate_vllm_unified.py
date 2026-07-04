#!/usr/bin/env python3
"""
Unified vLLM evaluation for scanpath prediction.

Consolidates grid-based and assume-normalized evaluation modes
into a single script. Supports base models, LoRA-finetuned models (merged),
and QLoRA models (native LoRA).

Metric modes:
  - grid: Full 100x100 probability grid via digit-by-digit logprob probing.
      Computes IG, AUC, NSS, LL.  (~1111 queries per transition)
  - assume_normalized: Scores only GT coordinates, assumes Z≈1 (or uses fixed
      --log-z).  Computes IG, LL.  Fastest mode.

Usage:
    # Zero-shot grid evaluation (finetuned model)
    python evaluate_vllm_unified.py \
        --base-model OpenGVLab/InternVL3_5-8B-HF \
        --adapter-path outputs/checkpoint-1000 \
        --val-json /path/to/val.json \
        --images-dir /path/to/images \
        --metric-mode grid --num-shots 0

    # Fast assume-normalized evaluation
    python evaluate_vllm_unified.py \
        --base-model model_path \
        --val-json /path/to/val.json \
        --images-dir /path/to/images \
        --assume-normalized --log-z -0.04
"""

import argparse
import json
import os
import pickle
import random
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# HuggingFace login for gated models (optional):
#   export HF_TOKEN=<your token>   # picked up automatically from the environment
# Never hardcode access tokens in source.


# =============================================================================
# Coordinate Formatting (separate-digits only)
# =============================================================================

def format_coordinate_reduced(value: float) -> str:
    """Format a single coordinate as zero-padded integer for v2-reduced format."""
    value = max(0.0, min(99.0, value))
    return f"{int(round(value)):02d}"


def format_scanpath_reduced(fixations: List[Tuple[float, float]]) -> str:
    """Format a list of fixations to v2-reduced scanpath string (zero-padded)."""
    parts = []
    for x, y in fixations:
        x_str = format_coordinate_reduced(x)
        y_str = format_coordinate_reduced(y)
        parts.append(f"({x_str}, {y_str})")
    return "[" + ", ".join(parts) + "]"


def format_partial_scanpath_reduced(
    fixations,
    partial_next: Optional[str] = None,
    xy_separator: str = ", ",
    temporal: bool = False,
    durations: bool = False,
) -> str:
    """Format partial scanpath for teacher forcing (zero-padded).

    Args:
        fixations: List of (x, y), (x, y, t), or (x, y, d) tuples.
        xy_separator: Separator between x and y within a coordinate pair.
            Auto-detected by SaliencyComputer (', ' or ',').
        temporal: If True, format completed fixations as (x, y, t) with
            4-digit zero-padded timestamps.
        durations: If True, format completed fixations as (x, y, d) with
            3-digit zero-padded durations.
    """
    if not fixations:
        if partial_next:
            return "[(" + partial_next
        else:
            return "[("

    parts = []
    for fix in fixations:
        x_str = format_coordinate_reduced(fix[0])
        y_str = format_coordinate_reduced(fix[1])
        if temporal and len(fix) >= 3:
            t_str = format_timestamp_reduced(fix[2])
            parts.append(f"({x_str}{xy_separator}{y_str}{xy_separator}{t_str})")
        elif durations and len(fix) >= 3:
            d_str = format_duration_reduced(fix[2])
            parts.append(f"({x_str}{xy_separator}{y_str}{xy_separator}{d_str})")
        else:
            parts.append(f"({x_str}{xy_separator}{y_str})")

    base = "[" + ", ".join(parts)

    if partial_next:
        return base + ", (" + partial_next
    else:
        return base + ", ("


def parse_scanpath_reduced(scanpath_str: str) -> List[Tuple[int, int]]:
    """Parse scanpath string into list of (x, y) tuples.

    Handles both 2-tuple (x, y) and 3-tuple (x, y, t) formats,
    always returning spatial-only (x, y) pairs.
    Coordinates are clipped to [0, 99].
    """
    # Try 3-tuple first, then fall back to 2-tuple
    pattern_3 = r'\((\d+),\s*(\d+),\s*\d+\)'
    matches = re.findall(pattern_3, scanpath_str)
    if matches:
        return [(min(int(x), 99), min(int(y), 99)) for x, y in matches]
    pattern_2 = r'\((\d+),\s*(\d+)\)'
    matches = re.findall(pattern_2, scanpath_str)
    return [(min(int(x), 99), min(int(y), 99)) for x, y in matches]


def parse_scanpath_temporal(scanpath_str: str) -> List[Tuple[int, int, int]]:
    """Parse temporal scanpath string into list of (x, y, t) tuples.

    Spatial coordinates are clipped to [0, 99].
    """
    pattern = r'\((\d+),\s*(\d+),\s*(\d+)\)'
    matches = re.findall(pattern, scanpath_str)
    return [(min(int(x), 99), min(int(y), 99), int(t)) for x, y, t in matches]


def detect_temporal_format(val_data: List[Dict], num_check: int = 5) -> bool:
    """Check if val data contains temporal (x, y, t) tuples with 4-digit timestamps."""
    for sample in val_data[:num_check]:
        gt_str = sample['conversations'][1]['value']
        matches = re.findall(r'\((\d+),\s*(\d+),\s*(\d+)\)', gt_str)
        if matches:
            # Temporal format uses 4-digit zero-padded timestamps
            third_lens = [len(m[2]) for m in matches]
            if all(l == 4 for l in third_lens):
                return True
    return False


def detect_durations_format(val_data: List[Dict], num_check: int = 5) -> bool:
    """Check if val data contains duration (x, y, d) tuples with 3-digit durations."""
    for sample in val_data[:num_check]:
        gt_str = sample['conversations'][1]['value']
        matches = re.findall(r'\((\d+),\s*(\d+),\s*(\d+)\)', gt_str)
        if matches:
            # Duration format uses 3-digit zero-padded durations
            third_lens = [len(m[2]) for m in matches]
            if all(l == 3 for l in third_lens):
                return True
    return False


def format_timestamp_reduced(t_ms: int) -> str:
    """Format timestamp as 4-digit zero-padded string."""
    t_ms = max(0, min(9999, t_ms))
    return f"{t_ms:04d}"


def format_duration_reduced(d_ms: int) -> str:
    """Format fixation duration as 3-digit zero-padded string."""
    d_ms = max(0, min(999, d_ms))
    return f"{d_ms:03d}"


# =============================================================================
# Digit Logprob Helpers
# =============================================================================

def _extract_digit_logprobs(logprobs_dict: Dict[str, float]) -> Dict[int, float]:
    """Extract digit 0-9 logprobs, filling missing with min of all logprobs.

    When max_logprobs=20 doesn't capture all 10 digits, missing digits are
    assigned min(all_logprobs) — an overestimate of the true (lower) logprob.
    This makes the logsumexp normalizer conservative (guarantees underestimation
    of normalized probabilities rather than overestimation).
    """
    digit_lps = {}
    for token, lp in logprobs_dict.items():
        if token and len(token) == 1 and token in '0123456789':
            digit_lps[int(token)] = lp
    if len(digit_lps) < 10:
        fill_val = min(logprobs_dict.values()) if logprobs_dict else -20.0
        for d in range(10):
            if d not in digit_lps:
                digit_lps[d] = fill_val
    return digit_lps


def _digit_logsumexp(logprobs_dict: Dict[str, float]) -> float:
    """Compute logsumexp over digit 0-9 logprobs from a raw logprobs dict.

    Uses _extract_digit_logprobs to fill missing digits conservatively.
    """
    digit_lps = _extract_digit_logprobs(logprobs_dict)
    return float(logsumexp(list(digit_lps.values())))


# =============================================================================
# Utility Functions
# =============================================================================

def extract_dataset_from_image_path(image_path: str) -> Optional[str]:
    """Extract dataset name from image path like 'images/MIT_0989.jpg'."""
    basename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.rsplit('_', 1)
    if len(parts) != 2:
        return None
    dataset_name = parts[0]
    known = {'MIT', 'CAT', 'CAT2000', 'COCO', 'Daemons', 'Figrim'}
    return dataset_name if dataset_name in known else None


# =============================================================================
# Subject Index (for subjective shot strategy)
# =============================================================================

DATASET_MAP = {
    'MIT': 'MIT', 'CAT': 'CAT', 'CAT2000': 'CAT',
    'COCO': 'COCO', 'Daemons': 'Daemons', 'Figrim': 'Figrim',
}


def _image_to_pkl_path(image_name: str, pkl_dir: str) -> str:
    """Convert image name (e.g. 'images/MIT_0989.jpg') to pkl file path."""
    basename = os.path.basename(image_name)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse image name '{image_name}' into (dataset, index)")
    dataset_name, index_str = parts
    dataset = DATASET_MAP.get(dataset_name)
    if dataset is None:
        raise ValueError(f"Unknown dataset '{dataset_name}' from image '{image_name}'")
    return os.path.join(pkl_dir, dataset, f"{index_str}.pkl")


def _load_pkl_subject_scanpaths(pkl_path: str) -> Dict[int, List[Tuple[int, int]]]:
    """Load pkl and return {subject_id: [(x_int, y_int), ...]} in v2-reduced coords.

    Uses the same 2-step conversion as convert_to_v2_scanpath.py:
        1. round(x / width * 100, 1)  ->  float with 1 decimal
        2. int(round(float))           ->  integer
    This avoids floating-point rounding mismatches vs 1-step int(round(x/w*100)).
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    img_h, img_w = data['image'].shape[1], data['image'].shape[2]
    subjects = data['subjects']
    fx = data['fixations_x']
    fy = data['fixations_y']

    result = {}
    for subj_id in np.unique(subjects):
        mask = subjects == subj_id
        sx = fx[mask]
        sy = fy[mask]
        scanpath = [
            (int(round(float(round((xi / img_w) * 100, 1)))),
             int(round(float(round((yi / img_h) * 100, 1)))))
            for xi, yi in zip(sx, sy)
        ]
        result[int(subj_id)] = scanpath
    return result


def resolve_sample_subject(
    sample: Dict, pkl_dir: str, _pkl_cache: Dict[str, Dict] = {}
) -> Tuple[str, int]:
    """Resolve a JSON sample to its (dataset, subject_id).

    Matches the sample's scanpath against pkl subject scanpaths.
    Raises ValueError if no match is found.
    """
    image_name = sample['images'][0]
    pkl_path = _image_to_pkl_path(image_name, pkl_dir)

    # Cache pkl subject scanpaths per image
    if pkl_path not in _pkl_cache:
        _pkl_cache[pkl_path] = _load_pkl_subject_scanpaths(pkl_path)
    subj_scanpaths = _pkl_cache[pkl_path]

    # Parse JSON scanpath (handle temporal 3-tuples by stripping timestamps)
    gt_str = sample['conversations'][1]['value']
    temporal_parsed = parse_scanpath_temporal(gt_str)
    if temporal_parsed:
        json_scanpath = [(x, y) for x, y, t in temporal_parsed]
    else:
        json_scanpath = parse_scanpath_reduced(gt_str)

    # Match against pkl subjects
    for subj_id, pkl_scanpath in subj_scanpaths.items():
        if pkl_scanpath == json_scanpath:
            dataset = extract_dataset_from_image_path(image_name)
            return (dataset, subj_id)
        elif pkl_scanpath[1:] == json_scanpath:  # Allow off-by-one for first fixation
            dataset = extract_dataset_from_image_path(image_name)
            warnings.warn(f"JSON Scanpath matched PKL subject with off-by-one (skipping first fixation). Image: {image_name}, Subject: {subj_id}. The model was likely trained without the first fixation, which is common in scanpath prediction. This sample will be matched to this subject AFTER REMOVING THE FIRST FIXATION, but please verify that the JSON and PKL scanpaths are consistent.")
            pkl_scanpath = pkl_scanpath[1:]  # Skip first fixation for matching
            return (dataset, subj_id)

    raise ValueError(
        f"Could not resolve subject for image '{image_name}': "
        f"JSON scanpath ({len(json_scanpath)} fixations, first={json_scanpath[0] if json_scanpath else '?'}) "
        f"does not match any of {len(subj_scanpaths)} subjects in {pkl_path}"
    )


def build_subject_index(
    samples: List[Dict], pkl_dir: str, label: str = "pool"
) -> Tuple[Dict[int, Tuple[str, int]], Dict[Tuple[str, int], List[int]]]:
    """Build subject index for a list of samples.

    Args:
        samples: List of JSON samples (train pool or val data)
        pkl_dir: Path to pickle files directory
        label: Label for progress messages ("pool" or "val")

    Returns:
        sample_to_subject: {sample_index: (dataset, subject_id)}
        subject_to_samples: {(dataset, subject_id): [sample_index, ...]}

    Raises ValueError if any sample cannot be resolved.
    """
    print(f"\nBuilding subject index for {label} ({len(samples)} samples)...")
    sample_to_subject: Dict[int, Tuple[str, int]] = {}
    subject_to_samples: Dict[Tuple[str, int], List[int]] = {}

    for i, sample in enumerate(samples):
        ds_subj = resolve_sample_subject(sample, pkl_dir)
        sample_to_subject[i] = ds_subj
        sample['_subject'] = ds_subj  # Annotate for fast lookup in ShotSelector
        if ds_subj not in subject_to_samples:
            subject_to_samples[ds_subj] = []
        subject_to_samples[ds_subj].append(i)

    n_subjects = len(subject_to_samples)
    print(f"  Resolved {len(samples)} samples to {n_subjects} unique subjects")
    for ds_subj, indices in sorted(subject_to_samples.items(), key=lambda x: (-len(x[1]), x[0])):
        if len(indices) >= 5:
            print(f"    {ds_subj}: {len(indices)} samples")

    return sample_to_subject, subject_to_samples


# =============================================================================
# Shot Selection
# =============================================================================

class ShotSelector:
    """Selects few-shot examples from a training pool."""

    def __init__(self, seed: int = 42, subject_index=None):
        self.rng = random.Random(seed)
        self._fixed_shots = None  # Cache for random-fixed strategy
        self.subject_index = subject_index

    def select(
        self,
        pool: List[Dict],
        num_shots: int,
        strategy: str = 'random',
        test_sample: Optional[Dict] = None,
    ) -> List[Dict]:
        """Select few-shot examples from pool."""
        if num_shots == 0:
            return []

        # random-fixed: select from full pool (no per-sample exclusion) and cache
        if strategy == 'random-fixed':
            if self._fixed_shots is not None:
                return self._fixed_shots
            candidates = list(pool)
            if len(candidates) == 0:
                return []
            return self._select_random_fixed(candidates, num_shots)

        # Filter out the test image if provided
        if test_sample is not None:
            test_images = set(test_sample.get('images', []))
            candidates = [s for s in pool if not set(s.get('images', [])) & test_images]
        else:
            candidates = list(pool)

        if len(candidates) == 0:
            return []

        if strategy == 'random':
            return self._select_random(candidates, num_shots)
        elif strategy == 'same-dataset':
            return self._select_same_dataset(candidates, num_shots, test_sample)
        elif strategy == 'diverse':
            return self._select_diverse(candidates, num_shots)
        elif strategy == 'subjective':
            return self._select_subjective(pool, num_shots, test_sample)
        else:
            raise ValueError(f"Unknown shot selection strategy: {strategy}")

    def _select_random(self, candidates: List[Dict], num_shots: int) -> List[Dict]:
        """Select random examples, ensuring unique images."""
        by_image = {}
        for s in candidates:
            img = s['images'][0] if s.get('images') else 'unknown'
            if img not in by_image:
                by_image[img] = s
        unique = list(by_image.values())
        n = min(num_shots, len(unique))
        return self.rng.sample(unique, n)

    def _select_random_fixed(self, candidates: List[Dict], num_shots: int) -> List[Dict]:
        """Select random examples once and reuse for all samples."""
        if self._fixed_shots is None:
            self._fixed_shots = self._select_random(candidates, num_shots)
        return self._fixed_shots

    def _select_same_dataset(
        self, candidates: List[Dict], num_shots: int, test_sample: Optional[Dict]
    ) -> List[Dict]:
        """Select examples from the same dataset as the test image."""
        if test_sample is None:
            return self._select_random(candidates, num_shots)

        test_dataset = None
        if test_sample.get('images'):
            test_dataset = extract_dataset_from_image_path(test_sample['images'][0])

        if test_dataset is None:
            return self._select_random(candidates, num_shots)

        same_ds = [
            s for s in candidates
            if s.get('images') and extract_dataset_from_image_path(s['images'][0]) == test_dataset
        ]

        by_image = {}
        for s in same_ds:
            img = s['images'][0]
            if img not in by_image:
                by_image[img] = s
        same_ds_unique = list(by_image.values())

        if len(same_ds_unique) >= num_shots:
            return self.rng.sample(same_ds_unique, num_shots)

        selected = list(same_ds_unique)
        remaining_candidates = [s for s in candidates if s not in selected]
        by_image_remaining = {}
        for s in remaining_candidates:
            img = s['images'][0] if s.get('images') else 'unknown'
            if img not in by_image_remaining and img not in by_image:
                by_image_remaining[img] = s
        remaining_unique = list(by_image_remaining.values())

        n_extra = min(num_shots - len(selected), len(remaining_unique))
        if n_extra > 0:
            selected.extend(self.rng.sample(remaining_unique, n_extra))

        return selected

    def _select_subjective(
        self, pool: List[Dict], num_shots: int, test_sample: Optional[Dict]
    ) -> List[Dict]:
        """Select random examples from the same subject as the test sample."""
        if self.subject_index is None:
            raise ValueError(
                "subjective strategy requires subject_index. "
                "Pass --pkl-dir to enable subject resolution."
            )

        pool_sample_to_subject, pool_subject_to_samples, val_sample_to_subject = self.subject_index

        test_images = set(test_sample.get('images', []))
        test_subject = test_sample.get('_subject')
        if test_subject is None:
            raise ValueError(
                f"Test sample for image '{test_sample.get('images', ['?'])[0]}' "
                f"has no _subject annotation. Did build_subject_index run on val data?"
            )

        if test_subject not in pool_subject_to_samples:
            raise ValueError(
                f"Subject {test_subject} has no training samples in the pool. "
                f"Image: {test_sample.get('images', ['?'])[0]}"
            )

        pool_indices = pool_subject_to_samples[test_subject]

        candidates = [
            pool[i] for i in pool_indices
            if not set(pool[i].get('images', [])) & test_images
        ]

        if len(candidates) < num_shots:
            raise ValueError(
                f"Subject {test_subject} has only {len(candidates)} training samples "
                f"(after excluding test image), need {num_shots}. "
                f"Image: {test_sample.get('images', ['?'])[0]}"
            )

        by_image = {}
        for s in candidates:
            img = s['images'][0] if s.get('images') else 'unknown'
            n_fix = len(parse_scanpath_reduced(s['conversations'][1]['value']))
            if img not in by_image or n_fix > by_image[img][1]:
                by_image[img] = (s, n_fix)
        unique = [s for s, _ in by_image.values()]

        if len(unique) < num_shots:
            raise ValueError(
                f"Subject {test_subject} has only {len(unique)} unique images "
                f"in training pool, need {num_shots}. "
                f"Image: {test_sample.get('images', ['?'])[0]}"
            )

        return self.rng.sample(unique, num_shots)

    def _select_diverse(self, candidates: List[Dict], num_shots: int) -> List[Dict]:
        """Select examples from diverse datasets (round-robin)."""
        by_dataset: Dict[str, List[Dict]] = {}
        for s in candidates:
            ds = None
            if s.get('images'):
                ds = extract_dataset_from_image_path(s['images'][0])
            if ds is None:
                ds = 'unknown'
            if ds not in by_dataset:
                by_dataset[ds] = []
            by_dataset[ds].append(s)

        for ds in by_dataset:
            by_image = {}
            for s in by_dataset[ds]:
                img = s['images'][0] if s.get('images') else 'unknown'
                if img not in by_image:
                    by_image[img] = s
            by_dataset[ds] = list(by_image.values())
            self.rng.shuffle(by_dataset[ds])

        selected = []
        ds_names = sorted(by_dataset.keys())
        idx = 0
        while len(selected) < num_shots:
            ds = ds_names[idx % len(ds_names)]
            if by_dataset[ds]:
                selected.append(by_dataset[ds].pop(0))
            else:
                ds_names.remove(ds)
                if not ds_names:
                    break
                continue
            idx += 1

        return selected


# =============================================================================
# Few-Shot Prompt Builder
# =============================================================================

class FewShotPromptBuilder:
    """Builds multi-turn prompts for few-shot scanpath evaluation."""

    def __init__(self, processor):
        self.processor = processor

    def build_prompt(
        self,
        test_image,
        test_prompt: str,
        shot_examples: List[Dict],
        partial_response: str = "",
    ) -> Tuple[str, Dict]:
        """Build a multi-turn prompt with few-shot examples.

        Args:
            test_image: PIL Image for the test case
            test_prompt: Text prompt for the test image
            shot_examples: List of dicts with 'image', 'prompt', 'response' keys
            partial_response: Partial response to append (for teacher forcing)

        Returns:
            Tuple of (formatted_prompt_string, mm_data_dict)
        """
        messages = []
        images = []

        for ex in shot_examples:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex['prompt']}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ex['response']}
                ]
            })
            images.append(ex['image'])

        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": test_prompt}
            ]
        })
        images.append(test_image)

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if len(images) == 1:
            mm_data = {"image": images[0]}
        else:
            mm_data = {"image": images}

        return prompt + partial_response, mm_data


def prepare_shot_examples(
    samples: List[Dict],
    images_dir: str,
) -> List[Dict]:
    """Convert LlamaFactory samples to shot examples with loaded images."""
    result = []
    for sample in samples:
        conv = sample['conversations']
        prompt = conv[0]['value'].replace("<image>", "").strip()
        response = conv[1]['value']

        image_path = os.path.join(images_dir, sample['images'][0])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load shot image {image_path}: {e}")
            continue

        result.append({
            'image': image,
            'prompt': prompt,
            'response': response,
        })

    return result


# =============================================================================
# Saliency Computer (unified: grid + MC scoring)
# =============================================================================

class SaliencyComputer:
    """Computes spatial saliency maps using vLLM with few-shot support.

    Supports two evaluation modes:
    - Grid mode: Full 100x100 probability grid via 4-phase digit-by-digit probing
    - MC/assume_normalized mode: Score specific coordinates via digit-by-digit probing
    """

    def __init__(
        self,
        llm,
        processor=None,
        prompt_builder: Optional[FewShotPromptBuilder] = None,
        lora_request=None,
        normalize_digits: bool = False,
    ):
        self.llm = llm
        self.processor = processor
        self.prompt_builder = prompt_builder
        self.lora_request = lora_request
        self.normalize_digits = normalize_digits
        self._xy_separator = None  # Auto-detected: ", " or ","

        if processor is None:
            try:
                from transformers import AutoProcessor
                model_path = llm.llm_engine.model_config.model
                self.processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: Could not load processor: {e}")

        self.tokenizer = self.processor.tokenizer if self.processor else None

        if self.prompt_builder is None and self.processor is not None:
            self.prompt_builder = FewShotPromptBuilder(self.processor)

        self.format_partial_scanpath = format_partial_scanpath_reduced
        self.format_coordinate = format_coordinate_reduced
        print("Using REDUCED token format (zero-padded, separate digits)")

    def build_prompt(
        self,
        image: Image.Image,
        text_prompt: str,
        shot_examples: List[Dict],
        partial_response: str = "",
    ) -> Tuple[str, Dict]:
        """Build prompt with optional few-shot examples."""
        return self.prompt_builder.build_prompt(
            test_image=image,
            test_prompt=text_prompt,
            shot_examples=shot_examples,
            partial_response=partial_response,
        )

    def detect_xy_separator(
        self,
        base_prompt: str,
        mm_data: Optional[Dict],
        best_x_d1: int = 5,
        best_x_d2: int = 0,
    ) -> str:
        """Auto-detect whether the model uses ', ' or ',' between x and y.

        Probes after the two x digits to check if the model predicts a comma,
        then probes after the comma to see if the next token is a space or digit.
        """
        if self._xy_separator is not None:
            return self._xy_separator

        # Probe after x digits: base_prompt + "50" -> should predict ","
        after_x = base_prompt + f"{best_x_d1}{best_x_d2}"
        comma_logprobs = self.generate_with_logprobs(after_x, mm_data, max_logprobs=20)

        # Check if ", " appears as a single token (some tokenizers merge them)
        if ", " in comma_logprobs:
            self._xy_separator = ", "
            print("  Detected xy separator: ', ' (merged token)")
            return self._xy_separator

        # Probe after comma: base_prompt + "50," -> space or digit?
        after_comma = after_x + ","
        space_logprobs = self.generate_with_logprobs(after_comma, mm_data, max_logprobs=20)

        space_logprob = space_logprobs.get(" ", -100.0)
        max_digit_logprob = -100.0
        for token, lp in space_logprobs.items():
            if token and len(token) == 1 and token in '0123456789':
                max_digit_logprob = max(max_digit_logprob, lp)

        if max_digit_logprob > space_logprob:
            self._xy_separator = ","
            print("  Detected xy separator: ',' (no space)")
        else:
            self._xy_separator = ", "
            print("  Detected xy separator: ', ' (with space)")

        return self._xy_separator

    def generate_with_logprobs(
        self,
        prompt: str,
        mm_data: Optional[Dict] = None,
        max_logprobs: int = 20,
    ) -> Dict[str, float]:
        """Generate one token and get logprobs."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=1,
            logprobs=max_logprobs,
            temperature=0.0,
        )

        if mm_data is not None:
            inputs = [{"prompt": prompt, "multi_modal_data": mm_data}]
        else:
            inputs = [prompt]

        kwargs = {}
        if self.lora_request is not None:
            kwargs['lora_request'] = self.lora_request

        outputs = self.llm.generate(inputs, sampling_params, **kwargs)

        result = {}
        if outputs and outputs[0].outputs[0].logprobs:
            for token_id, logprob_obj in outputs[0].outputs[0].logprobs[0].items():
                decoded = getattr(logprob_obj, 'decoded_token', str(token_id))
                result[decoded] = logprob_obj.logprob

        return result

    def generate_batched_with_logprobs(
        self,
        prompts: List[str],
        mm_data: Optional[Dict] = None,
        max_logprobs: int = 20,
    ) -> List[Dict[str, float]]:
        """Generate one token for multiple prompts (all share same mm_data)."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=1,
            logprobs=max_logprobs,
            temperature=0.0,
        )

        if mm_data is not None:
            inputs = [{"prompt": p, "multi_modal_data": mm_data} for p in prompts]
        else:
            inputs = prompts

        kwargs = {}
        if self.lora_request is not None:
            kwargs['lora_request'] = self.lora_request

        outputs = self.llm.generate(inputs, sampling_params, **kwargs)

        results = []
        for output in outputs:
            result = {}
            if output.outputs[0].logprobs:
                for token_id, logprob_obj in output.outputs[0].logprobs[0].items():
                    decoded = getattr(logprob_obj, 'decoded_token', str(token_id))
                    result[decoded] = logprob_obj.logprob
            results.append(result)

        return results

    def generate_batched_greedy(
        self,
        prompts: List[str],
        mm_data: Optional[Dict] = None,
        max_tokens: int = 3,
    ) -> List[str]:
        """Greedy-decode multiple tokens for each prompt. Returns generated text."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if mm_data is not None:
            inputs = [{"prompt": p, "multi_modal_data": mm_data} for p in prompts]
        else:
            inputs = prompts

        kwargs = {}
        if self.lora_request is not None:
            kwargs['lora_request'] = self.lora_request

        outputs = self.llm.generate(inputs, sampling_params, **kwargs)
        return [output.outputs[0].text for output in outputs]

    # =========================================================================
    # Grid mode: full 100x100 probability grids
    # =========================================================================

    def compute_next_fixation_distribution(
        self,
        image: Image.Image,
        prompt: str,
        shot_examples: List[Dict],
        previous_fixations: List[Tuple[int, int]],
        resolution: int = 100,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Compute 2D probability distribution for next fixation (single transition)."""
        xy_sep = self._xy_separator or ", "
        ASCII_DIGITS = set('0123456789')
        partial = self.format_partial_scanpath(previous_fixations, xy_separator=xy_sep)
        base_prompt, mm_data = self.build_prompt(image, prompt, shot_examples, partial)

        grid = np.full((resolution, resolution), -20.0)

        # Phase 1: P(x_d1)
        d1_logprobs = self.generate_with_logprobs(base_prompt, mm_data)
        d1_raw = {}
        for token, logprob in d1_logprobs.items():
            if token and len(token) == 1 and token in ASCII_DIGITS:
                d1_raw[int(token)] = logprob

        if not d1_raw:
            return grid

        # Phase 2: P(x_d2 | x_d1)
        p2_prompts = []
        p2_d1 = []
        for d1, raw_lp in d1_raw.items():
            p2_prompts.append(base_prompt + str(d1))
            p2_d1.append(d1)

        x_logprobs = {}
        if p2_prompts:
            p2_results = self.generate_batched_with_logprobs(p2_prompts, mm_data)
            for d1, d2_logprobs in zip(p2_d1, p2_results):
                for d2 in range(10):
                    d2_str = str(d2)
                    if d2_str in d2_logprobs:
                        x_int = d1 * 10 + d2
                        if x_int < resolution:
                            x_logprobs[x_int] = d1_raw[d1] + d2_logprobs[d2_str]

        if not x_logprobs:
            return grid

        # Auto-detect separator
        best_x = max(x_logprobs, key=x_logprobs.get)
        xy_sep = self.detect_xy_separator(base_prompt, mm_data, best_x // 10, best_x % 10)

        # Phase 3: P(y_d1 | x)
        p3_prompts = []
        p3_x = []
        p3_log_px = []
        for x_int, log_p_x in x_logprobs.items():
            d1 = x_int // 10
            d2 = x_int % 10
            p3_prompts.append(base_prompt + f"{d1}{d2}{xy_sep}")
            p3_x.append(x_int)
            p3_log_px.append(log_p_x)

        if not p3_prompts:
            return grid

        p3_results = self.generate_batched_with_logprobs(p3_prompts, mm_data)

        # Phase 4: P(y_d2 | x, y_d1)
        p4_prompts = []
        p4_info = []

        for x_int, log_p_x, x_prompt, yd1_logprobs in zip(p3_x, p3_log_px, p3_prompts, p3_results):
            yd1_raw = {}
            for token, logprob in yd1_logprobs.items():
                if token and len(token) == 1 and token in ASCII_DIGITS:
                    yd1_raw[int(token)] = logprob

            if not yd1_raw:
                continue

            for y_d1, raw_lp in yd1_raw.items():
                p4_prompts.append(x_prompt + str(y_d1))
                p4_info.append((x_int, log_p_x, y_d1, raw_lp))

        for i in range(0, len(p4_prompts), batch_size):
            batch_prompts = p4_prompts[i:i+batch_size]
            batch_info = p4_info[i:i+batch_size]
            batch_results = self.generate_batched_with_logprobs(batch_prompts, mm_data)

            for (x_int, log_p_x, y_d1, log_p_yd1), yd2_logprobs in zip(batch_info, batch_results):
                for token, logprob in yd2_logprobs.items():
                    if token and len(token) == 1 and token in ASCII_DIGITS:
                        y_d2 = int(token)
                        y_int = y_d1 * 10 + y_d2
                        if y_int < resolution:
                            grid[y_int, x_int] = log_p_x + log_p_yd1 + logprob

        return grid

    def compute_all_fixation_distributions(
        self,
        image: Image.Image,
        prompt: str,
        shot_examples: List[Dict],
        gt_fixations: List[Tuple[int, int]],
        resolution: int = 100,
        batch_size: int = 256,
    ) -> List[np.ndarray]:
        """Compute distributions for ALL fixation transitions in one batched pass."""
        n_transitions = len(gt_fixations) - 1
        if n_transitions <= 0:
            return []

        return self._compute_all_distributions_separate_digits(
            image, prompt, shot_examples, gt_fixations, resolution, batch_size
        )

    def compute_multi_scanpath_distributions(
        self,
        image: Image.Image,
        text_prompts: List[str],
        shot_examples: List[Dict],
        all_gt_fixations: List[List[Tuple[int, int]]],
        resolution: int = 100,
        batch_size: int = 256,
    ) -> List[List[np.ndarray]]:
        """Compute distributions for ALL scanpaths' transitions in one batched pass.

        Groups all transitions from all scanpaths into single batched vLLM calls,
        maximizing prefix cache reuse for the shared image tokens.
        """
        if len(all_gt_fixations) == 0:
            return []

        valid = [(tp, gf) for tp, gf in zip(text_prompts, all_gt_fixations)
                 if len(gf) >= 2]
        if not valid:
            return [[] for _ in all_gt_fixations]

        if len(valid) == 1 and len(all_gt_fixations) == 1:
            tp, gf = valid[0]
            grids = self.compute_all_fixation_distributions(
                image, tp, shot_examples, gf, resolution, batch_size
            )
            return [grids]

        return self._compute_multi_distributions_separate_digits(
            image, text_prompts, shot_examples, all_gt_fixations,
            resolution, batch_size
        )

    def _compute_multi_distributions_separate_digits(
        self,
        image: Image.Image,
        text_prompts: List[str],
        shot_examples: List[Dict],
        all_gt_fixations: List[List[Tuple[int, int]]],
        resolution: int = 100,
        batch_size: int = 256,
    ) -> List[List[np.ndarray]]:
        """Batched 4-phase computation across ALL scanpaths' transitions."""
        xy_sep = self._xy_separator or ", "
        ASCII_DIGITS = set('0123456789')

        # Build flat base_prompts for ALL scanpaths' transitions
        base_prompts = []
        scanpath_offsets = []
        mm_data = None
        offset = 0

        for text_prompt, gt_fix in zip(text_prompts, all_gt_fixations):
            n_trans = max(0, len(gt_fix) - 1)
            scanpath_offsets.append((offset, n_trans))
            for i in range(1, len(gt_fix)):
                previous = gt_fix[:i]
                partial = self.format_partial_scanpath(previous, xy_separator=xy_sep)
                bp, md = self.build_prompt(image, text_prompt, shot_examples, partial)
                base_prompts.append(bp)
                if mm_data is None:
                    mm_data = md
            offset += n_trans

        total_transitions = offset
        if total_transitions == 0:
            return [[] for _ in all_gt_fixations]

        grids = [np.full((resolution, resolution), -20.0) for _ in range(total_transitions)]

        # Phase 1: Batch ALL P(x_d1) queries
        p1_results = self.generate_batched_with_logprobs(base_prompts, mm_data, max_logprobs=20)

        per_t_d1 = []
        for t, d1_logprobs in enumerate(p1_results):
            d1_raw = {}
            for token, logprob in d1_logprobs.items():
                if token and len(token) == 1 and token in ASCII_DIGITS:
                    d1_raw[int(token)] = logprob
            per_t_d1.append(d1_raw)

        # Phase 2: Batch ALL P(x_d2 | x_d1) queries
        p2_prompts = []
        p2_index = []

        for t in range(total_transitions):
            d1_raw = per_t_d1[t]
            if not d1_raw:
                continue
            for d1, raw_lp in d1_raw.items():
                p2_prompts.append(base_prompts[t] + str(d1))
                p2_index.append((t, d1))

        per_t_x = [dict() for _ in range(total_transitions)]

        if p2_prompts:
            p2_results = []
            for i in range(0, len(p2_prompts), batch_size):
                batch = p2_prompts[i:i+batch_size]
                p2_results.extend(self.generate_batched_with_logprobs(batch, mm_data, max_logprobs=20))

            for (t, d1), d2_logprobs in zip(p2_index, p2_results):
                for d2 in range(10):
                    d2_str = str(d2)
                    if d2_str in d2_logprobs:
                        x_int = d1 * 10 + d2
                        if x_int < resolution:
                            per_t_x[t][x_int] = per_t_d1[t][d1] + d2_logprobs[d2_str]

        # Auto-detect separator once
        for t in range(total_transitions):
            if per_t_x[t]:
                best_x = max(per_t_x[t], key=per_t_x[t].get)
                xy_sep = self.detect_xy_separator(
                    base_prompts[t], mm_data, best_x // 10, best_x % 10
                )
                break

        # Phase 3: Batch ALL P(y_d1 | x) queries
        p3_prompts = []
        p3_index = []

        for t in range(total_transitions):
            x_logprobs = per_t_x[t]
            if not x_logprobs:
                continue
            for x_int, log_p_x in x_logprobs.items():
                d1 = x_int // 10
                d2 = x_int % 10
                x_suffix = f"{d1}{d2}{xy_sep}"
                p3_prompts.append(base_prompts[t] + x_suffix)
                p3_index.append((t, x_int, log_p_x))

        if not p3_prompts:
            result = []
            for start, n_trans in scanpath_offsets:
                result.append(grids[start:start + n_trans])
            return result

        p3_results = []
        for i in range(0, len(p3_prompts), batch_size):
            batch = p3_prompts[i:i+batch_size]
            p3_results.extend(self.generate_batched_with_logprobs(batch, mm_data, max_logprobs=20))

        # Phase 4: Batch ALL P(y_d2 | x, y_d1) queries
        p4_prompts = []
        p4_info = []

        for (t, x_int, log_p_x), x_prompt, y_d1_logprobs in zip(p3_index, p3_prompts, p3_results):
            yd1_raw = {}
            for token, logprob in y_d1_logprobs.items():
                if token and len(token) == 1 and token in ASCII_DIGITS:
                    yd1_raw[int(token)] = logprob

            if not yd1_raw:
                continue

            for y_d1, raw_lp in yd1_raw.items():
                p4_prompts.append(x_prompt + str(y_d1))
                p4_info.append((t, x_int, log_p_x, y_d1, raw_lp))

        for i in range(0, len(p4_prompts), batch_size):
            batch_prompts = p4_prompts[i:i+batch_size]
            batch_info = p4_info[i:i+batch_size]
            batch_results = self.generate_batched_with_logprobs(batch_prompts, mm_data, max_logprobs=20)

            for (t, x_int, log_p_x, y_d1, log_p_y_d1), y_d2_logprobs in zip(batch_info, batch_results):
                for token, logprob in y_d2_logprobs.items():
                    if token and len(token) == 1 and token in ASCII_DIGITS:
                        y_d2 = int(token)
                        y_int = y_d1 * 10 + y_d2
                        if y_int < resolution:
                            total_log_p = log_p_x + log_p_y_d1 + logprob
                            grids[t][y_int, x_int] = total_log_p

        # Split flat grids back to per-scanpath
        result = []
        for start, n_trans in scanpath_offsets:
            result.append(grids[start:start + n_trans])
        return result

    def _compute_all_distributions_separate_digits(
        self,
        image: Image.Image,
        prompt: str,
        shot_examples: List[Dict],
        gt_fixations: List[Tuple[int, int]],
        resolution: int = 100,
        batch_size: int = 256,
    ) -> List[np.ndarray]:
        """Batched 4-phase computation across all fixation transitions (single scanpath)."""
        return self._compute_multi_distributions_separate_digits(
            image, [prompt], shot_examples, [gt_fixations],
            resolution, batch_size,
        )[0]

    # =========================================================================
    # MC/assume_normalized mode: score specific coordinates
    # =========================================================================

    def generate_model_samples(
        self,
        base_prompts: List[str],
        mm_data: Optional[Dict],
        num_samples: int,
        xy_sep: str,
        batch_size: int,
        temperature: float = 1.0,
    ) -> List[List[Tuple[int, int]]]:
        """Generate coordinate samples from the model's distribution.

        For each base prompt, generates num_samples coordinate strings via
        temperature sampling. Returns parsed (x, y) coordinates.

        Args:
            base_prompts: One prompt per transition (ending with '(')
            mm_data: Multimodal data dict (shared across all prompts)
            num_samples: Number of samples to generate per transition
            xy_sep: Separator between x and y
            batch_size: vLLM batch size
            temperature: Sampling temperature (1.0 = model distribution)

        Returns:
            List of T lists, each containing up to num_samples (x, y) tuples.
        """
        import re

        from vllm import SamplingParams

        T = len(base_prompts)
        # max_tokens: "07, 52)" = up to 7 tokens for separate digit models
        max_tokens = 7

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=num_samples,
            logprobs=0,
        )

        # Build all inputs
        all_inputs = []
        for prompt in base_prompts:
            if mm_data is not None:
                all_inputs.append({"prompt": prompt, "multi_modal_data": mm_data})
            else:
                all_inputs.append(prompt)

        kwargs = {}
        if self.lora_request is not None:
            kwargs['lora_request'] = self.lora_request

        # Generate in batches
        all_outputs = []
        for i in range(0, len(all_inputs), batch_size):
            batch = all_inputs[i:i + batch_size]
            all_outputs.extend(
                self.llm.generate(batch, sampling_params, **kwargs)
            )

        # Parse generated coordinates
        coord_pattern = re.compile(r'^(\d{2})' + re.escape(xy_sep) + r'(\d{2})')
        per_transition_coords = []
        for output in all_outputs:
            coords = []
            for seq_output in output.outputs:
                text = seq_output.text.strip()
                m = coord_pattern.match(text)
                if m:
                    x = int(m.group(1))
                    y = int(m.group(2))
                    if 0 <= x < 100 and 0 <= y < 100:
                        coords.append((x, y))
            per_transition_coords.append(coords)

        return per_transition_coords

    def score_coordinates(
        self,
        base_prompts: List[str],
        mm_data: Optional[Dict],
        transition_info: List[Dict],
        mc_coords: List[Tuple[int, int]],
        xy_sep: str,
        batch_size: int,
    ) -> List[Dict[Tuple[int, int], float]]:
        """4-phase digit-by-digit scoring for specific coordinates.

        For each transition, scores all MC coordinates + the GT coordinate.

        Args:
            base_prompts: One prompt per transition (with partial scanpath appended)
            mm_data: Multimodal data dict (shared across all prompts)
            transition_info: List of dicts with 'gt_target' key per transition
            mc_coords: List of (x, y) MC sample coordinates to score
            xy_sep: Separator between x and y (', ' or ',')
            batch_size: vLLM batch size for chunking

        Returns:
            List of T dicts, each mapping (x, y) -> log P(coord | base)
        """
        ASCII_DIGITS = set('0123456789')
        T = len(base_prompts)
        K = len(mc_coords)

        # Precompute digit decomposition of MC coordinates
        mc_x_d1 = [c[0] // 10 for c in mc_coords]
        mc_x_d2 = [c[0] % 10 for c in mc_coords]
        mc_x = [c[0] for c in mc_coords]
        mc_y_d1 = [c[1] // 10 for c in mc_coords]
        mc_y_d2 = [c[1] % 10 for c in mc_coords]

        unique_mc_xd1 = sorted(set(mc_x_d1))
        unique_mc_x = sorted(set(mc_x))
        unique_mc_x_yd1 = sorted(set(zip(mc_x, mc_y_d1)))

        # Phase 1: P(x_d1 | base) for all T transitions
        print(f"      Phase 1: {T} prompts (x first digit)", flush=True)
        p1_results = []
        for i in range(0, T, batch_size):
            batch = base_prompts[i:i + batch_size]
            p1_results.extend(
                self.generate_batched_with_logprobs(batch, mm_data)
            )

        per_t_xd1_lp = []
        for result in p1_results:
            d_lps = {}
            for token, lp in result.items():
                if token and len(token) == 1 and token in ASCII_DIGITS:
                    d_lps[int(token)] = lp
            per_t_xd1_lp.append(d_lps)

        # Phase 2: P(x_d2 | base, x_d1)
        p2_prompts = []
        p2_index = []

        for t in range(T):
            gt_xd1 = transition_info[t]['gt_target'][0] // 10
            needed_xd1 = set(unique_mc_xd1)
            needed_xd1.add(gt_xd1)
            for d1 in sorted(needed_xd1):
                if d1 in per_t_xd1_lp[t]:
                    p2_prompts.append(base_prompts[t] + str(d1))
                    p2_index.append((t, d1))

        print(f"      Phase 2: {len(p2_prompts)} prompts (x second digit)", flush=True)
        p2_results = []
        for i in range(0, len(p2_prompts), batch_size):
            batch = p2_prompts[i:i + batch_size]
            p2_results.extend(
                self.generate_batched_with_logprobs(batch, mm_data)
            )

        per_t_xd2_lp = [dict() for _ in range(T)]
        for (t, d1), result in zip(p2_index, p2_results):
            for token, lp in result.items():
                if token and len(token) == 1 and token in ASCII_DIGITS:
                    per_t_xd2_lp[t][(d1, int(token))] = lp

        # Phase 3: P(y_d1 | base, x, sep)
        p3_prompts = []
        p3_index = []

        for t in range(T):
            gt_x = transition_info[t]['gt_target'][0]
            needed_x = set(unique_mc_x)
            needed_x.add(gt_x)
            for x_val in sorted(needed_x):
                d1 = x_val // 10
                d2 = x_val % 10
                if d1 in per_t_xd1_lp[t] and (d1, d2) in per_t_xd2_lp[t]:
                    x_str = f"{d1}{d2}{xy_sep}"
                    p3_prompts.append(base_prompts[t] + x_str)
                    p3_index.append((t, x_val))

        print(f"      Phase 3: {len(p3_prompts)} prompts (y first digit)", flush=True)
        p3_results = []
        for i in range(0, len(p3_prompts), batch_size):
            batch = p3_prompts[i:i + batch_size]
            p3_results.extend(
                self.generate_batched_with_logprobs(batch, mm_data)
            )

        per_t_yd1_lp = [dict() for _ in range(T)]
        for (t, x_val), result in zip(p3_index, p3_results):
            for token, lp in result.items():
                if token and len(token) == 1 and token in ASCII_DIGITS:
                    per_t_yd1_lp[t][(x_val, int(token))] = lp

        # Phase 4: P(y_d2 | base, x, sep, y_d1)
        p4_prompts = []
        p4_index = []

        for t in range(T):
            gt_x, gt_y = transition_info[t]['gt_target']
            gt_yd1 = gt_y // 10
            needed_x_yd1 = set(unique_mc_x_yd1)
            needed_x_yd1.add((gt_x, gt_yd1))
            for x_val, yd1 in sorted(needed_x_yd1):
                if (x_val, yd1) in per_t_yd1_lp[t]:
                    d1 = x_val // 10
                    d2 = x_val % 10
                    prompt = base_prompts[t] + f"{d1}{d2}{xy_sep}{yd1}"
                    p4_prompts.append(prompt)
                    p4_index.append((t, x_val, yd1))

        print(f"      Phase 4: {len(p4_prompts)} prompts (y second digit)", flush=True)
        p4_results = []
        for i in range(0, len(p4_prompts), batch_size):
            batch = p4_prompts[i:i + batch_size]
            p4_results.extend(
                self.generate_batched_with_logprobs(batch, mm_data)
            )

        per_t_yd2_lp = [dict() for _ in range(T)]
        for (t, x_val, yd1), result in zip(p4_index, p4_results):
            for token, lp in result.items():
                if token and len(token) == 1 and token in ASCII_DIGITS:
                    per_t_yd2_lp[t][(x_val, yd1, int(token))] = lp

        # Compute digit normalizers (logsumexp over 10 digits at each phase)
        z1 = {}  # z1[t]
        z2 = {}  # z2[(t, d1)]
        z3 = {}  # z3[(t, x_val)]
        z4 = {}  # z4[(t, x_val, yd1)]

        if self.normalize_digits:
            for t in range(T):
                z1[t] = _digit_logsumexp(p1_results[t])
            for (t, d1), result in zip(p2_index, p2_results):
                z2[(t, d1)] = _digit_logsumexp(result)
            for (t, x_val), result in zip(p3_index, p3_results):
                z3[(t, x_val)] = _digit_logsumexp(result)
            for (t, x_val, yd1), result in zip(p4_index, p4_results):
                z4[(t, x_val, yd1)] = _digit_logsumexp(result)

        # Assemble log P(coord | base) for each transition
        per_transition_lls = []
        for t in range(T):
            coord_lls = {}
            gt_target = transition_info[t]['gt_target']
            all_coords = list(mc_coords) + [gt_target]

            for coord in all_coords:
                x, y = coord
                d1x = x // 10
                d2x = x % 10
                d1y = y // 10
                d2y = y % 10

                lp_xd1 = per_t_xd1_lp[t].get(d1x, -20.0)
                lp_xd2 = per_t_xd2_lp[t].get((d1x, d2x), -20.0)
                lp_yd1 = per_t_yd1_lp[t].get((x, d1y), -20.0)
                lp_yd2 = per_t_yd2_lp[t].get((x, d1y, d2y), -20.0)

                if self.normalize_digits:
                    lp_xd1 -= z1.get(t, 0.0)
                    lp_xd2 -= z2.get((t, d1x), 0.0)
                    lp_yd1 -= z3.get((t, x), 0.0)
                    lp_yd2 -= z4.get((t, x, d1y), 0.0)

                coord_lls[coord] = lp_xd1 + lp_xd2 + lp_yd1 + lp_yd2

            per_transition_lls.append(coord_lls)

        return per_transition_lls

    def score_temporal_gt(
        self,
        base_prompts: List[str],
        mm_data,
        transition_info: List[Dict],
        xy_sep: str,
        batch_size: int,
    ) -> List[float]:
        """Score GT timestamp log-likelihood via 4-digit autoregressive probing.

        For each transition, builds prompt ending with GT (x, y) coordinates,
        then probes the 4 GT timestamp digits sequentially.

        Args:
            base_prompts: Prompts ending with '(' (same as for score_coordinates).
            mm_data: Shared multimodal data (image).
            transition_info: Must contain 'gt_target' (x, y) and 'gt_timestamp' (int ms).
            xy_sep: Separator string (', ' or ',').
            batch_size: vLLM batch size.

        Returns:
            List of temporal log-likelihoods, one per transition.
        """
        T = len(base_prompts)

        # Build temporal base prompts: base + "x_d1 x_d2 sep y_d1 y_d2 sep"
        # This conditions on GT (x, y) and prepares for timestamp probing
        temporal_bases = []
        gt_t_digits = []  # List of 4-digit lists for each transition
        for t in range(T):
            x, y = transition_info[t]['gt_target']
            t_ms = transition_info[t]['gt_timestamp']
            x_str = format_coordinate_reduced(x)
            y_str = format_coordinate_reduced(y)
            temporal_bases.append(base_prompts[t] + f"{x_str}{xy_sep}{y_str}{xy_sep}")
            t_str = format_timestamp_reduced(t_ms)
            gt_t_digits.append([int(d) for d in t_str])

        temporal_lls = [0.0] * T

        # Phase T1: P(t_d1 | base, x, y)
        print(f"      Temporal Phase 1: {T} prompts (t first digit)", flush=True)
        t1_results = []
        for i in range(0, T, batch_size):
            batch = temporal_bases[i:i + batch_size]
            t1_results.extend(self.generate_batched_with_logprobs(batch, mm_data))
        for t in range(T):
            d = str(gt_t_digits[t][0])
            lp = t1_results[t].get(d, -20.0)
            if self.normalize_digits:
                lp -= _digit_logsumexp(t1_results[t])
            temporal_lls[t] += lp

        # Phase T2: P(t_d2 | base, x, y, t_d1)
        t2_prompts = [temporal_bases[t] + str(gt_t_digits[t][0]) for t in range(T)]
        print(f"      Temporal Phase 2: {T} prompts (t second digit)", flush=True)
        t2_results = []
        for i in range(0, T, batch_size):
            batch = t2_prompts[i:i + batch_size]
            t2_results.extend(self.generate_batched_with_logprobs(batch, mm_data))
        for t in range(T):
            d = str(gt_t_digits[t][1])
            lp = t2_results[t].get(d, -20.0)
            if self.normalize_digits:
                lp -= _digit_logsumexp(t2_results[t])
            temporal_lls[t] += lp

        # Phase T3: P(t_d3 | base, x, y, t_d1, t_d2)
        t3_prompts = [
            temporal_bases[t] + str(gt_t_digits[t][0]) + str(gt_t_digits[t][1])
            for t in range(T)
        ]
        print(f"      Temporal Phase 3: {T} prompts (t third digit)", flush=True)
        t3_results = []
        for i in range(0, T, batch_size):
            batch = t3_prompts[i:i + batch_size]
            t3_results.extend(self.generate_batched_with_logprobs(batch, mm_data))
        for t in range(T):
            d = str(gt_t_digits[t][2])
            lp = t3_results[t].get(d, -20.0)
            if self.normalize_digits:
                lp -= _digit_logsumexp(t3_results[t])
            temporal_lls[t] += lp

        # Phase T4: P(t_d4 | base, x, y, t_d1, t_d2, t_d3)
        t4_prompts = [
            temporal_bases[t] + str(gt_t_digits[t][0]) + str(gt_t_digits[t][1]) + str(gt_t_digits[t][2])
            for t in range(T)
        ]
        print(f"      Temporal Phase 4: {T} prompts (t fourth digit)", flush=True)
        t4_results = []
        for i in range(0, T, batch_size):
            batch = t4_prompts[i:i + batch_size]
            t4_results.extend(self.generate_batched_with_logprobs(batch, mm_data))
        for t in range(T):
            d = str(gt_t_digits[t][3])
            lp = t4_results[t].get(d, -20.0)
            if self.normalize_digits:
                lp -= _digit_logsumexp(t4_results[t])
            temporal_lls[t] += lp

        return temporal_lls

    def score_duration_gt(
        self,
        base_prompts: List[str],
        mm_data,
        transition_info: List[Dict],
        xy_sep: str,
        batch_size: int,
    ) -> List[float]:
        """Score GT duration log-likelihood via 3-digit autoregressive probing.

        For each transition, builds prompt ending with GT (x, y) coordinates,
        then probes the 3 GT duration digits sequentially.

        Args:
            base_prompts: Prompts ending with '(' (same as for score_coordinates).
            mm_data: Shared multimodal data (image).
            transition_info: Must contain 'gt_target' (x, y) and 'gt_duration' (int ms).
            xy_sep: Separator string (', ' or ',').
            batch_size: vLLM batch size.

        Returns:
            List of duration log-likelihoods, one per transition.
        """
        T = len(base_prompts)

        # Build duration base prompts: base + "x_d1 x_d2 sep y_d1 y_d2 sep"
        # This conditions on GT (x, y) and prepares for duration probing
        duration_bases = []
        gt_d_digits = []  # List of 3-digit lists for each transition
        for t in range(T):
            x, y = transition_info[t]['gt_target']
            d_ms = transition_info[t]['gt_duration']
            x_str = format_coordinate_reduced(x)
            y_str = format_coordinate_reduced(y)
            duration_bases.append(base_prompts[t] + f"{x_str}{xy_sep}{y_str}{xy_sep}")
            d_str = format_duration_reduced(d_ms)
            gt_d_digits.append([int(c) for c in d_str])

        duration_lls = [0.0] * T

        # Phase D1: P(d_d1 | base, x, y)
        print(f"      Duration Phase 1: {T} prompts (d first digit)", flush=True)
        d1_results = []
        for i in range(0, T, batch_size):
            batch = duration_bases[i:i + batch_size]
            d1_results.extend(self.generate_batched_with_logprobs(batch, mm_data))
        for t in range(T):
            d = str(gt_d_digits[t][0])
            lp = d1_results[t].get(d, -20.0)
            if self.normalize_digits:
                lp -= _digit_logsumexp(d1_results[t])
            duration_lls[t] += lp

        # Phase D2: P(d_d2 | base, x, y, d_d1)
        d2_prompts = [duration_bases[t] + str(gt_d_digits[t][0]) for t in range(T)]
        print(f"      Duration Phase 2: {T} prompts (d second digit)", flush=True)
        d2_results = []
        for i in range(0, T, batch_size):
            batch = d2_prompts[i:i + batch_size]
            d2_results.extend(self.generate_batched_with_logprobs(batch, mm_data))
        for t in range(T):
            d = str(gt_d_digits[t][1])
            lp = d2_results[t].get(d, -20.0)
            if self.normalize_digits:
                lp -= _digit_logsumexp(d2_results[t])
            duration_lls[t] += lp

        # Phase D3: P(d_d3 | base, x, y, d_d1, d_d2)
        d3_prompts = [
            duration_bases[t] + str(gt_d_digits[t][0]) + str(gt_d_digits[t][1])
            for t in range(T)
        ]
        print(f"      Duration Phase 3: {T} prompts (d third digit)", flush=True)
        d3_results = []
        for i in range(0, T, batch_size):
            batch = d3_prompts[i:i + batch_size]
            d3_results.extend(self.generate_batched_with_logprobs(batch, mm_data))
        for t in range(T):
            d = str(gt_d_digits[t][2])
            lp = d3_results[t].get(d, -20.0)
            if self.normalize_digits:
                lp -= _digit_logsumexp(d3_results[t])
            duration_lls[t] += lp

        return duration_lls

    def predict_duration_greedy(
        self,
        base_prompts: List[str],
        mm_data,
        transition_info: List[Dict],
        xy_sep: str,
        batch_size: int,
    ) -> List[int]:
        """Greedy-decode the 3-digit duration for each transition in a single vLLM call.

        Conditions on GT (x, y), then generates 3 tokens greedily.
        Parses the output as a duration in ms.

        Returns:
            List of predicted durations in ms, one per transition.
        """
        T = len(base_prompts)

        # Build duration base prompts: base + "x_str sep y_str sep"
        duration_bases = []
        for t in range(T):
            x, y = transition_info[t]['gt_target']
            x_str = format_coordinate_reduced(x)
            y_str = format_coordinate_reduced(y)
            duration_bases.append(base_prompts[t] + f"{x_str}{xy_sep}{y_str}{xy_sep}")

        # Single batched greedy generation: 3 tokens per prompt
        print(f"      Duration greedy: {T} prompts (3 tokens each)", flush=True)
        generated = []
        for i in range(0, T, batch_size):
            batch = duration_bases[i:i + batch_size]
            generated.extend(self.generate_batched_greedy(batch, mm_data, max_tokens=3))

        # Parse 3-digit durations from generated text
        predicted = []
        n_fallback = 0
        for t in range(T):
            text = generated[t]
            # Extract up to 3 leading digits
            digits = ''.join(c for c in text[:3] if c.isdigit())
            if len(digits) >= 3:
                predicted.append(int(digits[:3]))
            elif digits:
                # Partial digits (model stopped early) — pad with 0
                predicted.append(int(digits.ljust(3, '0')))
                n_fallback += 1
            else:
                predicted.append(0)  # Fallback
                n_fallback += 1
        if n_fallback > 0:
            print(f"        WARNING: {n_fallback}/{T} transitions had <3 digit tokens "
                  f"(first raw output: {generated[0]!r})", flush=True)
        return predicted


# =============================================================================
# Centerbias Functions
# =============================================================================

def create_center_bias(shape: Tuple[int, int] = (100, 100), sigma: float = 20.0) -> np.ndarray:
    """Create Gaussian center bias log-density map."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    log_density = -((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)
    log_density = log_density - logsumexp(log_density)
    return log_density


def load_centerbias_from_pkl(
    image_name: str, pkl_dir: str, resolution: int = 100
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Load centerbias from pkl file and resize to evaluation grid."""
    basename = os.path.basename(image_name)
    name_without_ext = os.path.splitext(basename)[0]

    parts = name_without_ext.rsplit('_', 1)
    if len(parts) != 2:
        return None, None

    dataset_name, index_str = parts

    dataset_map = {
        'MIT': 'MIT', 'CAT': 'CAT', 'CAT2000': 'CAT',
        'COCO': 'COCO', 'Daemons': 'Daemons', 'Figrim': 'Figrim',
    }
    dataset = dataset_map.get(dataset_name, dataset_name)

    pkl_path = os.path.join(pkl_dir, dataset, f"{index_str}.pkl")

    if not os.path.exists(pkl_path):
        return None, None

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        centerbias = np.array(data['centerbias'])

        density = np.exp(centerbias - centerbias.max())
        zoom_y = resolution / density.shape[0]
        zoom_x = resolution / density.shape[1]
        density_resized = zoom(density, (zoom_y, zoom_x), order=1)
        density_resized = density_resized / density_resized.sum()
        density_resized = np.clip(density_resized, 1e-10, None)
        log_density_resized = np.log(density_resized)

        return log_density_resized, pkl_path

    except Exception as e:
        print(f"Warning: Failed to load centerbias from {pkl_path}: {e}")
        return None, None


# =============================================================================
# Centerbias Alpha Tuning
# =============================================================================

ALPHA_CANDIDATES = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 6.0, 7.5, 9.0, 10.0]


def tune_centerbias_alpha_grid(
    saliency_computer: SaliencyComputer,
    tune_samples: List[Dict],
    images_dir: str,
    shot_pool: List[Dict],
    shot_selector: ShotSelector,
    num_shots: int,
    shot_strategy: str,
    pkl_dir: Optional[str],
    default_center_bias: np.ndarray,
    use_data_centerbias: bool,
    resolution: int,
    batch_size: int,
    alpha_candidates: List[float] = ALPHA_CANDIDATES,
) -> float:
    """Tune centerbias alpha using full grid computation (for grid metric mode)."""
    import time as _time

    print(f"\n{'='*70}")
    print("Tuning centerbias alpha (grid mode)")
    print(f"{'='*70}")
    print(f"  Tuning samples: {len(tune_samples)}")
    print(f"  Alpha candidates: {alpha_candidates}")
    tune_start = _time.time()

    fixation_data = []
    samples_processed = 0

    for idx, sample in enumerate(tune_samples):
        image_path = os.path.join(images_dir, sample['images'][0])

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"  [{idx+1}/{len(tune_samples)}] SKIP {sample['images'][0]}: {e}")
            continue

        conv = sample['conversations']
        text_prompt = conv[0]['value'].replace("<image>", "").strip()
        gt_fixations = parse_scanpath_reduced(conv[1]['value'])

        if len(gt_fixations) < 2:
            continue

        if use_data_centerbias and pkl_dir:
            cb, _ = load_centerbias_from_pkl(sample['images'][0], pkl_dir, resolution)
            if cb is None:
                cb = default_center_bias
        else:
            cb = default_center_bias

        selected_shots = shot_selector.select(
            shot_pool, num_shots, shot_strategy, test_sample=sample
        )
        shot_examples = prepare_shot_examples(selected_shots, images_dir)

        n_transitions = len(gt_fixations) - 1
        print(f"  [{idx+1}/{len(tune_samples)}] {sample['images'][0]}: "
              f"{len(gt_fixations)} fixations ({n_transitions} transitions)",
              flush=True)

        all_grids = saliency_computer.compute_all_fixation_distributions(
            image, text_prompt, shot_examples, gt_fixations, resolution, batch_size
        )

        for i, model_grid in enumerate(all_grids):
            target = gt_fixations[i + 1]
            fixation_data.append({
                'model_grid': model_grid,
                'centerbias': cb,
                'target': target,
            })

        samples_processed += 1

    if not fixation_data:
        print("  No valid tuning data, using alpha=0.0")
        return 0.0

    print(f"\n  Collected {len(fixation_data)} transitions from {samples_processed} samples")

    for fd in fixation_data:
        fd['model_log_prob'] = fd['model_grid'] - logsumexp(fd['model_grid'])

    best_alpha = 0.0
    best_mean_ll = -np.inf

    for alpha in alpha_candidates:
        lls = []
        for fd in fixation_data:
            combined = fd['model_log_prob'] + alpha * fd['centerbias']
            combined = combined - logsumexp(combined)
            x, y = fd['target']
            x_idx = min(resolution - 1, max(0, x))
            y_idx = min(resolution - 1, max(0, y))
            lls.append(float(combined[y_idx, x_idx]))

        mean_ll = np.mean(lls)
        marker = " <-- best" if mean_ll > best_mean_ll else ""
        print(f"    alpha={alpha:5.1f}: mean LL = {mean_ll:.4f}{marker}")

        if mean_ll > best_mean_ll:
            best_mean_ll = mean_ll
            best_alpha = alpha

    total_elapsed = _time.time() - tune_start
    print(f"\n  Best alpha: {best_alpha} (mean LL = {best_mean_ll:.4f})")
    print(f"  Total tuning time: {total_elapsed:.1f}s")
    print(f"{'='*70}\n")
    return best_alpha


# =============================================================================
# Metrics (for grid mode)
# =============================================================================

def compute_information_gain(
    model_log_density: np.ndarray,
    baseline_log_density: np.ndarray,
    fixation: Tuple[int, int],
    resolution: int = 100,
) -> float:
    """Compute Information Gain at a fixation location."""
    x, y = fixation
    x_idx = min(resolution - 1, max(0, x))
    y_idx = min(resolution - 1, max(0, y))
    model_log_p = model_log_density[y_idx, x_idx]
    baseline_log_p = baseline_log_density[y_idx, x_idx]
    ig = (model_log_p - baseline_log_p) / np.log(2)
    return ig


def compute_auc(
    log_density: np.ndarray,
    fixation: Tuple[int, int],
    resolution: int = 100,
) -> float:
    """Compute AUC for a saliency map at a fixation.

    Uses all pixels as negatives (matching deepgaze-iccv/pysaliency).
    """
    x, y = fixation
    x_idx = min(resolution - 1, max(0, x))
    y_idx = min(resolution - 1, max(0, y))
    fixation_value = log_density[y_idx, x_idx]
    all_values = log_density.flatten()
    auc = np.mean(fixation_value > all_values) + 0.5 * np.mean(fixation_value == all_values)
    return auc


def compute_nss(
    log_density: np.ndarray,
    fixation: Tuple[int, int],
    resolution: int = 100,
) -> float:
    """Compute Normalized Scanpath Saliency (NSS)."""
    x, y = fixation
    x_idx = min(resolution - 1, max(0, x))
    y_idx = min(resolution - 1, max(0, y))
    density = np.exp(log_density - logsumexp(log_density))
    mean = density.mean()
    std = density.std()
    if std > 0:
        nss = (density[y_idx, x_idx] - mean) / std
    else:
        nss = 0.0
    return nss


def compute_log_nss(
    log_density: np.ndarray,
    fixation: Tuple[int, int],
    resolution: int = 100,
) -> float:
    """Compute log-space NSS: z-score of GT log-density vs all grid log-densities."""
    x, y = fixation
    x_idx = min(resolution - 1, max(0, x))
    y_idx = min(resolution - 1, max(0, y))
    log_vals = log_density.flatten()
    mean = log_vals.mean()
    std = log_vals.std()
    if std > 0:
        return float((log_density[y_idx, x_idx] - mean) / std)
    return 0.0


# =============================================================================
# Visualization (grid mode only)
# =============================================================================

def sample_scanpath_from_model(
    saliency_computer: SaliencyComputer,
    image: Image.Image,
    prompt: str,
    shot_examples: List[Dict],
    num_fixations: int,
    resolution: int = 100,
    temperature: float = 1.0,
) -> List[Tuple[int, int]]:
    """Sample a scanpath from the model by iteratively sampling fixations."""
    scanpath = []
    scanpath.append((50, 50))

    for _ in range(num_fixations - 1):
        log_density = saliency_computer.compute_next_fixation_distribution(
            image, prompt, shot_examples, scanpath, resolution, batch_size=256
        )

        valid_mask = np.isfinite(log_density)
        if not valid_mask.any():
            scanpath.append((50, 50))
            continue

        log_density[~valid_mask] = -np.inf
        log_density = log_density - logsumexp(log_density[valid_mask])

        if temperature != 1.0:
            log_density = log_density / temperature
            log_density = log_density - logsumexp(log_density[valid_mask])

        probs = np.exp(log_density)
        probs = probs / probs.sum()

        flat_idx = np.random.choice(resolution * resolution, p=probs.flatten())
        y = flat_idx // resolution
        x = flat_idx % resolution

        scanpath.append((x, y))

    return scanpath


def plot_scanpath_comparison(
    image_path: str,
    gt_scanpath: List[Tuple[int, int]],
    pred_scanpath: List[Tuple[int, int]],
    output_path: str,
    title: str = "",
    resolution: int = 100,
):
    """Plot GT and predicted scanpaths overlaid on the image."""
    image = Image.open(image_path).convert('RGB')
    img_w, img_h = image.size

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(image)

    def to_pixels(coords):
        return [(x / resolution * img_w, y / resolution * img_h) for x, y in coords]

    gt_pixels = to_pixels(gt_scanpath)
    pred_pixels = to_pixels(pred_scanpath)

    gt_xs = [p[0] for p in gt_pixels]
    gt_ys = [p[1] for p in gt_pixels]
    ax.plot(gt_xs, gt_ys, 'g-', linewidth=2, alpha=0.7, label='GT')
    ax.scatter(gt_xs, gt_ys, c='green', s=100, zorder=5, edgecolors='white', linewidths=2)
    for i, (x, y) in enumerate(gt_pixels):
        ax.annotate(str(i+1), (x, y), fontsize=10, ha='center', va='center',
                   color='white', fontweight='bold')

    pred_xs = [p[0] for p in pred_pixels]
    pred_ys = [p[1] for p in pred_pixels]
    ax.plot(pred_xs, pred_ys, 'r-', linewidth=2, alpha=0.7, label='Predicted')
    ax.scatter(pred_xs, pred_ys, c='red', s=100, zorder=5, edgecolors='white', linewidths=2)
    for i, (x, y) in enumerate(pred_pixels):
        ax.annotate(str(i+1), (x, y), fontsize=10, ha='center', va='center',
                   color='white', fontweight='bold')

    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_fixation_heatmaps(
    saliency_computer: SaliencyComputer,
    image: Image.Image,
    image_path: str,
    prompt: str,
    shot_examples: List[Dict],
    gt_scanpath: List[Tuple[int, int]],
    output_dir: str,
    sample_idx: int,
    resolution: int = 100,
    batch_size: int = 256,
):
    """Create heatmaps showing model's predicted distribution for each GT fixation."""
    img_w, img_h = image.size
    num_fixations = len(gt_scanpath)

    n_cols = min(4, num_fixations - 1)
    n_rows = (num_fixations - 2) // n_cols + 1

    if num_fixations <= 1:
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(1, num_fixations):
        row = (i - 1) // n_cols
        col = (i - 1) % n_cols
        ax = axes[row, col]

        previous = gt_scanpath[:i]
        target = gt_scanpath[i]

        log_density = saliency_computer.compute_next_fixation_distribution(
            image, prompt, shot_examples, previous, resolution, batch_size
        )

        valid_mask = np.isfinite(log_density)
        if valid_mask.any():
            log_density[~valid_mask] = log_density[valid_mask].min()
            log_density = log_density - logsumexp(log_density)

        prob_map = np.exp(log_density)
        prob_map_resized = zoom(prob_map, (img_h / resolution, img_w / resolution), order=1)

        ax.imshow(image)
        ax.imshow(prob_map_resized, alpha=0.5, cmap='hot')

        target_x = target[0] / resolution * img_w
        target_y = target[1] / resolution * img_h
        ax.scatter([target_x], [target_y], c='cyan', s=200, marker='*',
                  edgecolors='white', linewidths=2, zorder=10, label='GT')

        for j, (px, py) in enumerate(previous):
            px_img = px / resolution * img_w
            py_img = py / resolution * img_h
            ax.scatter([px_img], [py_img], c='green', s=80, edgecolors='white', linewidths=1)
            ax.annotate(str(j+1), (px_img, py_img), fontsize=8, ha='center', va='center',
                       color='white', fontweight='bold')

        ax.set_title(f'Fixation {i+1} (target: {target})', fontsize=10)
        ax.axis('off')

    for i in range(num_fixations - 1, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.suptitle(f'Sample {sample_idx}: Per-Fixation Heatmaps', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'sample_{sample_idx}_heatmaps.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_samples_by_ig(
    all_results: List[Dict],
    val_data: List[Dict],
    saliency_computer: SaliencyComputer,
    shot_pool: List[Dict],
    shot_selector: ShotSelector,
    num_shots: int,
    shot_strategy: str,
    images_dir: str,
    output_dir: str,
    num_samples: int = 20,
    resolution: int = 100,
):
    """Select samples evenly spaced by IG and create visualizations."""
    results_with_ig = [(r, i) for i, r in enumerate(all_results) if r.get('lp_mean_ig') is not None]

    if len(results_with_ig) < num_samples:
        num_samples = len(results_with_ig)

    if num_samples == 0:
        print("No samples with valid IG to visualize")
        return

    results_with_ig.sort(key=lambda x: x[0]['lp_mean_ig'])
    indices = np.linspace(0, len(results_with_ig) - 1, num_samples, dtype=int)
    selected = [results_with_ig[i] for i in indices]

    scanpath_dir = os.path.join(output_dir, 'scanpath_comparisons')
    heatmap_dir = os.path.join(output_dir, 'fixation_heatmaps')
    os.makedirs(scanpath_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    print(f"\nGenerating visualizations for {num_samples} samples...")

    for rank, (result, orig_idx) in enumerate(tqdm(selected)):
        sample = val_data[result['idx']]
        image_path = os.path.join(images_dir, sample['images'][0])

        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert('RGB')
        gt_scanpath = result['gt_fixations']
        ig = result['lp_mean_ig']

        conv = sample['conversations']
        prompt = conv[0]['value'].replace("<image>", "").strip()

        # Select shots for this image
        selected_shots = shot_selector.select(
            shot_pool, num_shots, shot_strategy, test_sample=sample
        )
        shot_examples = prepare_shot_examples(selected_shots, images_dir)

        pred_scanpath = sample_scanpath_from_model(
            saliency_computer, image, prompt, shot_examples,
            len(gt_scanpath), resolution, temperature=0.8
        )

        title = f"Sample {result['idx']} | IG={ig:.2f} | Rank {rank+1}/{num_samples}"
        scanpath_output = os.path.join(scanpath_dir, f'rank{rank+1:02d}_sample{result["idx"]}_ig{ig:.2f}.png')
        plot_scanpath_comparison(image_path, gt_scanpath, pred_scanpath, scanpath_output, title, resolution)

        plot_fixation_heatmaps(
            saliency_computer, image, image_path, prompt, shot_examples, gt_scanpath,
            heatmap_dir, result['idx'], resolution
        )


# =============================================================================
# Model Loading
# =============================================================================

def merge_lora_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into base model for vLLM compatibility."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    print("\n" + "="*70)
    print("MERGING LORA ADAPTER")
    print("="*70)
    print(f"  Base model: {base_model_path}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output: {output_path}")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        print("  Adapter config:")
        print(f"    - base_model: {adapter_config.get('base_model_name_or_path', 'N/A')}")
        print(f"    - r (rank): {adapter_config.get('r', 'N/A')}")
        print(f"    - lora_alpha: {adapter_config.get('lora_alpha', 'N/A')}")

    base_model_lower = base_model_path.lower()
    is_internvl = 'internvl' in base_model_lower
    is_smolvlm = 'smolvlm' in base_model_lower
    is_gemma3 = 'gemma-3' in base_model_lower or 'gemma3' in base_model_lower
    is_paligemma = 'paligemma' in base_model_lower
    is_llava = 'llava' in base_model_lower
    is_qwen = 'qwen' in base_model_lower

    print("\n  Loading base model on CPU (merge does not need GPU)...")
    if is_qwen:
        from transformers import AutoModelForImageTextToText
        print("  (Using AutoModelForImageTextToText for Qwen)")
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu"
        )
    elif is_gemma3:
        import transformers
        tf_version = tuple(map(int, transformers.__version__.split('.')[:2]))
        if tf_version < (5, 0):
            from transformers import AutoModelForImageTextToText
            print(f"  (Using AutoModelForImageTextToText for Gemma3, transformers {transformers.__version__})")
            model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="cpu"
            )
        else:
            from transformers import AutoModelForMultimodalLM
            print(f"  (Using AutoModelForMultimodalLM for Gemma3, transformers {transformers.__version__})")
            model = AutoModelForMultimodalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="cpu"
            )
    elif is_internvl or is_smolvlm or is_paligemma or is_llava:
        from transformers import AutoModelForImageTextToText
        model_type = "InternVL" if is_internvl else "SmolVLM" if is_smolvlm else "PaliGemma" if is_paligemma else "LLaVA"
        print(f"  (Using AutoModelForImageTextToText for {model_type})")
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu"
        )
    else:
        raise ValueError(f"Unsupported model type: {base_model_path}. Add support in merge_lora_adapter().")

    print("  Loading adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("  Merging weights...")
    model = model.merge_and_unload()

    print(f"  Saving to {output_path}...")
    model.save_pretrained(output_path)

    processor_kwargs = {"trust_remote_code": True}
    if is_gemma3:
        processor_kwargs["use_fast"] = False
    processor = AutoProcessor.from_pretrained(base_model_path, **processor_kwargs)
    processor.save_pretrained(output_path)

    if is_gemma3:
        config_path = os.path.join(output_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        patched = False
        rope_scaling_val = {"factor": 8.0, "rope_type": "linear"}
        if "rope_scaling" not in config:
            config["rope_scaling"] = rope_scaling_val
            patched = True
        if "text_config" in config and "rope_scaling" not in config["text_config"]:
            config["text_config"]["rope_scaling"] = rope_scaling_val
            patched = True
        if patched:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print("  Patched config.json with rope_scaling for vLLM compatibility")

    print("  Merged model saved!")

    # Free GPU memory before vLLM loads the merged model
    del model
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print("  Freed merge model from GPU memory")
    print("="*70 + "\n")
    return output_path


def resolve_adapter_path(adapter_path: str) -> Tuple[str, Dict]:
    """Resolve adapter path to a specific checkpoint directory.

    If adapter_path already points to a checkpoint-* directory, return as-is.
    If it points to a training output root (containing trainer_log.jsonl and
    checkpoint-* subdirectories), auto-detect the best checkpoint by minimum
    eval_loss from trainer_log.jsonl.

    Falls back to the latest checkpoint if no eval_loss entries exist.

    Returns:
        (resolved_path, info_dict) where info_dict contains resolution details
        for display in the startup banner.
    """
    info = {'method': 'explicit'}

    # Already a checkpoint directory
    if os.path.basename(adapter_path).startswith('checkpoint-'):
        return adapter_path, info

    # Check if this is a training output root with checkpoints
    import glob
    # Exclude _merged directories before parsing step numbers
    checkpoint_dirs = [
        d for d in glob.glob(os.path.join(adapter_path, 'checkpoint-[0-9]*'))
        if not d.endswith('_merged')
    ]
    checkpoint_dirs.sort(key=lambda d: int(os.path.basename(d).split('-')[1]))

    if not checkpoint_dirs:
        # Not a training root — maybe it's a direct model path, return as-is
        return adapter_path, info

    info['training_root'] = adapter_path
    available_steps = [int(os.path.basename(d).split('-')[1]) for d in checkpoint_dirs]
    info['available_checkpoints'] = available_steps

    # Try to find best checkpoint from trainer_log.jsonl
    trainer_log = os.path.join(adapter_path, 'trainer_log.jsonl')
    best_step = None

    if os.path.isfile(trainer_log):
        eval_entries = []
        with open(trainer_log, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'eval_loss' in entry:
                        eval_entries.append(entry)
                except json.JSONDecodeError:
                    continue

        if eval_entries:
            best_entry = min(eval_entries, key=lambda x: x['eval_loss'])
            best_step = best_entry['current_steps']
            info['method'] = 'best_eval_loss'
            info['best_eval_loss'] = best_entry['eval_loss']
            info['best_step'] = best_step

    if best_step is not None:
        valid_steps = [s for s in available_steps if s <= best_step]
        if valid_steps:
            chosen_step = max(valid_steps)
        else:
            chosen_step = min(available_steps)
            info['warning'] = f"No checkpoint <= step {best_step}, using earliest"
    else:
        chosen_step = max(available_steps)
        info['method'] = 'latest'
        if os.path.isfile(trainer_log):
            info['warning'] = "No eval_loss entries in trainer_log.jsonl"
        else:
            info['warning'] = "No trainer_log.jsonl found"

    info['chosen_step'] = chosen_step
    resolved = os.path.join(adapter_path, f'checkpoint-{chosen_step}')
    return resolved, info


def load_model_vllm(
    base_model: str,
    adapter_path: Optional[str] = None,
    merge_only: bool = False,
    use_native_lora: bool = False,
    quantization_bit: Optional[int] = None,
    num_images: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    enable_prefix_caching: bool = True,
    max_model_len: int = 16384,
    max_num_seqs: int = 256,
    enforce_eager: bool = False,
    max_lora_rank: int = 64,
):
    """Unified model loader supporting base, LoRA (merged), and QLoRA (native LoRA).

    Args:
        base_model: HuggingFace model name/path
        adapter_path: LoRA adapter checkpoint (optional, triggers merge)
        merge_only: Just merge adapter and exit
        use_native_lora: Use native LoRA instead of merging (for QLoRA)
        quantization_bit: Quantization bit width (for QLoRA)
        num_images: Max images per prompt (num_shots + 1)
        tensor_parallel_size: Number of GPUs
        gpu_memory_utilization: Fraction of GPU memory to use
        enable_prefix_caching: Enable vLLM prefix caching
        max_model_len: Maximum context length
        max_num_seqs: Maximum concurrent sequences
        enforce_eager: Disable CUDA graphs
        max_lora_rank: Maximum LoRA rank for native LoRA

    Returns:
        (llm, lora_request) tuple. lora_request is None unless use_native_lora.
    """
    from vllm import LLM

    lora_request = None

    if adapter_path and use_native_lora:
        # QLoRA: load base model with native LoRA support
        from vllm.lora.request import LoRARequest

        quantization = None
        if quantization_bit:
            quantization = "bitsandbytes" if quantization_bit in (4, 8) else None

        print("\n" + "="*70)
        print("LOADING MODEL WITH NATIVE LORA SUPPORT")
        print("="*70)
        print(f"  Base model: {base_model}")
        print(f"  Adapter path: {adapter_path}")
        print(f"  Quantization: {quantization or 'None'}")
        print(f"  Max LoRA rank: {max_lora_rank}")

        llm_kwargs = dict(
            model=base_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt={"image": num_images},
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
        )

        if quantization:
            llm_kwargs["quantization"] = quantization

        base_model_lower = base_model.lower()
        if 'gemma-3' in base_model_lower or 'gemma3' in base_model_lower:
            llm_kwargs["hf_config_path"] = base_model

        llm = LLM(**llm_kwargs)
        lora_request = LoRARequest(
            lora_name="finetuned_adapter",
            lora_int_id=1,
            lora_path=adapter_path,
        )
        print("vLLM model with native LoRA loaded successfully!")
        print("="*70 + "\n")
        return llm, lora_request

    if adapter_path:
        # LoRA: merge adapter into base model
        merged_path = f"{adapter_path}_merged"

        if os.path.isdir(merged_path):
            print(f"Using existing merged model: {merged_path}")
        else:
            merge_lora_adapter(base_model, adapter_path, merged_path)

        if merge_only:
            print("Merge complete. Exiting (--merge-only).")
            import sys
            sys.exit(0)

        model_path = merged_path
    else:
        # Base model: load directly
        model_path = base_model

    print("\nLoading model with vLLM...")
    print(f"  Model: {model_path}")
    print(f"  Max images per prompt: {num_images}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print(f"  Max model len: {max_model_len}")
    print(f"  Prefix caching: {enable_prefix_caching}")
    print(f"  Max num seqs: {max_num_seqs}")
    print(f"  Enforce eager: {enforce_eager}")

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=enable_prefix_caching,
        limit_mm_per_prompt={"image": num_images},
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
    )

    base_model_lower = base_model.lower()
    if 'gemma-3' in base_model_lower or 'gemma3' in base_model_lower:
        llm_kwargs["hf_config_path"] = base_model
        print(f"  Using hf_config_path={base_model} for Gemma3")

    llm = LLM(**llm_kwargs)
    print("vLLM model loaded successfully!")
    return llm, None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified vLLM Evaluation for Scanpath Prediction'
    )

    # Model
    parser.add_argument('--base-model', type=str, default='OpenGVLab/InternVL3_5-8B-HF',
                        help='Base model name (HuggingFace)')
    parser.add_argument('--adapter-path', type=str, default=None,
                        help='LoRA adapter checkpoint path (triggers merge)')
    parser.add_argument('--merge-only', action='store_true',
                        help='Just merge adapter and exit')
    parser.add_argument('--use-native-lora', action='store_true',
                        help='Use native LoRA instead of merging (for QLoRA)')
    parser.add_argument('--quantization-bit', type=int, default=None,
                        help='Quantization bit width (for QLoRA, e.g. 4)')

    # Data
    parser.add_argument('--val-json', type=str, required=True,
                        help='Path to validation JSON file (LlamaFactory format)')
    parser.add_argument('--images-dir', type=str, default='/mnt/lustre/work/bethge/bkr710/projects/lvlm-gaze/llamafactory_data_scanpath_v2_lodo',
                        help='Base directory for images')
    parser.add_argument('--output-dir', type=str, default='eval_unified_outputs')

    # Few-shot settings
    parser.add_argument('--num-shots', type=int, default=0,
                        help='Number of few-shot examples (0 = zero-shot)')
    parser.add_argument('--shot-pool-json', type=str, default=None,
                        help='JSON file with training samples for few-shot examples')
    parser.add_argument('--shot-pool-max', type=int, default=500,
                        help='Max samples to load from shot pool (for memory)')
    parser.add_argument('--shot-strategy', type=str, default='random',
                        choices=['random', 'random-fixed', 'diverse', 'same-dataset', 'subjective'],
                        help='Strategy for selecting few-shot examples')

    # Metric mode
    parser.add_argument('--metric-mode', type=str, default='assume_normalized',
                        choices=['grid', 'assume_normalized'],
                        help='assume_normalized (default, fast IG/LL with normalize-digits) '
                             'or grid (full 100x100)')
    parser.add_argument('--assume-normalized', action='store_true',
                        help='Shorthand for --metric-mode assume_normalized')
    parser.add_argument('--log-z', type=float, default=None,
                        help='Fixed log Z value (implies assume_normalized mode)')

    # Centerbias
    parser.add_argument('--pkl-dir', type=str, default='/mnt/lustre/work/bethge/bkr710/projects/deepgaze-iccv/tmp_datasets_withsubj',
                        help='Directory with pkl files for data-driven centerbias')
    parser.add_argument('--tune-centerbias', action='store_true',
                        help='Tune centerbias alpha via grid search')
    parser.add_argument('--centerbias-alpha', type=float, default=None,
                        help='Manually set centerbias alpha (skip tuning)')
    parser.add_argument('--tune-num-samples', type=int, default=20,
                        help='Number of samples for centerbias alpha tuning')

    # Infrastructure
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=384,
                        help='Number of prompts per vLLM call')
    parser.add_argument('--max-model-len', type=int, default=4096,
                        help='Max context length')
    parser.add_argument('--max-num-seqs', type=int, default=256,
                        help='Max concurrent sequences in vLLM')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.95)
    parser.add_argument('--enforce-eager', action='store_true', default=False,
                        help='Disable CUDA graphs')
    parser.add_argument('--no-enforce-eager', dest='enforce_eager', action='store_false')

    # Visualization
    parser.add_argument('--skip-viz', action='store_true')
    parser.add_argument('--num-viz-samples', type=int, default=10)
    parser.add_argument('--save-grids', action='store_true',
                        help='Save per-transition log-density grids to .npz files (grid mode only)')

    # Resume
    parser.add_argument('--resume-json', type=str, default=None,
                        help='Path to partial results JSON to resume from')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify model loading')
    parser.add_argument('--skip-temporal', action='store_true',
                        help='Skip temporal eval even if temporal data detected')
    parser.add_argument('--skip-durations', action='store_true',
                        help='Skip duration eval even if duration data detected')
    parser.add_argument('--temporal', action='store_true',
                        help='Force temporal mode (4-digit onset timestamps)')
    parser.add_argument('--durations', action='store_true',
                        help='Force durations mode (3-digit fixation durations)')
    parser.add_argument('--normalize-digits', action='store_true', default=True,
                        help='Normalize digit logprobs at each phase (condition on next token being a digit). '
                             'Makes log Z ~ 0. Enabled by default.')
    parser.add_argument('--no-normalize-digits', dest='normalize_digits', action='store_false',
                        help='Disable digit logprob normalization.')

    args = parser.parse_args()

    # Shorthands
    if args.assume_normalized:
        args.metric_mode = 'assume_normalized'
    if args.log_z is not None:
        args.metric_mode = 'assume_normalized'
    if args.normalize_digits:
        if args.log_z is not None and args.log_z != 0.0:
            print(f"WARNING: --normalize-digits makes --log-z redundant. "
                  f"Ignoring --log-z {args.log_z} and using log_z=0.")
        args.log_z = 0.0

    # Validate
    if args.temporal and args.durations:
        parser.error("--temporal and --durations are mutually exclusive")
    if args.num_shots > 0 and args.shot_pool_json is None:
        parser.error("--shot-pool-json is required when --num-shots > 0")
    if args.shot_strategy == 'subjective' and not args.pkl_dir:
        parser.error("--pkl-dir is required for subjective strategy")

    # Resolve adapter path (auto-detect best checkpoint if given a training root)
    _adapter_info = {}
    if args.adapter_path:
        args.adapter_path, _adapter_info = resolve_adapter_path(args.adapter_path)

    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load val data early for format detection in banner
    print(f"Loading validation data from {args.val_json}...")
    with open(args.val_json, 'r') as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples")

    # Detect data format: temporal, durations, or spatial-only
    if args.temporal:
        is_temporal = True
        is_durations = False
        _format_str = "TEMPORAL (forced via --temporal) - 4-digit onset timestamps"
    elif args.durations:
        is_temporal = False
        is_durations = True
        _format_str = "DURATIONS (forced via --durations) - 3-digit fixation durations"
    else:
        is_temporal = detect_temporal_format(val_data)
        is_durations = detect_durations_format(val_data) if not is_temporal else False
        if is_temporal:
            _format_str = "TEMPORAL (auto-detected) - 4-digit onset timestamps"
        elif is_durations:
            _format_str = "DURATIONS (auto-detected) - 3-digit fixation durations"
        else:
            _format_str = "SPATIAL-ONLY (auto-detected) - (x, y) tuples"

    # Handle skip flags
    if is_temporal and args.skip_temporal:
        _format_str += "  [SKIPPED via --skip-temporal]"
        is_temporal = False
    if is_durations and args.skip_durations:
        _format_str += "  [SKIPPED via --skip-durations]"
        is_durations = False

    print("=" * 70)
    print("Unified vLLM Scanpath Evaluation")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    if args.adapter_path:
        print(f"Adapter path: {args.adapter_path}")
        if _adapter_info.get('method') == 'best_eval_loss':
            print(f"  (auto-selected: best eval_loss={_adapter_info['best_eval_loss']:.6f} "
                  f"at step {_adapter_info['best_step']}, "
                  f"checkpoint-{_adapter_info['chosen_step']})")
        elif _adapter_info.get('method') == 'latest':
            print(f"  (auto-selected: latest checkpoint-{_adapter_info['chosen_step']})")
            if _adapter_info.get('warning'):
                print(f"  WARNING: {_adapter_info['warning']}")
    print(f"Num shots: {args.num_shots}")
    print(f"Shot strategy: {args.shot_strategy}")
    print(f"Metric mode: {args.metric_mode}")
    if args.metric_mode == 'assume_normalized':
        if args.log_z is not None:
            print(f"  (using fixed log Z = {args.log_z})")
        else:
            print("  (assuming Z~1, log Z=0)")
    if args.normalize_digits:
        print("Digit normalization: ENABLED (log Z should be ~ 0)")
    print(f"Val JSON: {args.val_json}")
    print(f"Images dir: {args.images_dir}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Data format: {_format_str}")
    print("=" * 70)

    # =========================================================================
    # Load model
    # =========================================================================
    num_images = args.num_shots + 1
    llm, lora_request = load_model_vllm(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        merge_only=args.merge_only,
        use_native_lora=args.use_native_lora,
        quantization_bit=args.quantization_bit,
        num_images=num_images,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )

    # Get processor
    from transformers import AutoProcessor
    # Use the actual loaded model path for processor (may be merged path)
    processor_model = args.base_model
    try:
        processor = AutoProcessor.from_pretrained(
            processor_model, trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: Could not load processor: {e}")
        processor = None

    # Create saliency computer
    prompt_builder = FewShotPromptBuilder(processor) if processor else None
    saliency_computer = SaliencyComputer(
        llm, processor,
        prompt_builder=prompt_builder,
        lora_request=lora_request,
        normalize_digits=args.normalize_digits,
    )

    # =========================================================================
    # Apply data limits
    # =========================================================================
    if args.max_samples:
        val_data = val_data[:args.max_samples]

    # Load shot pool if needed
    shot_pool = []
    if args.num_shots > 0:
        print(f"Loading shot pool from {args.shot_pool_json}...")
        with open(args.shot_pool_json, 'r') as f:
            shot_pool = json.load(f)
        if args.shot_strategy != 'subjective':
            if args.shot_pool_max and len(shot_pool) > args.shot_pool_max:
                random.shuffle(shot_pool)
                shot_pool = shot_pool[:args.shot_pool_max]
        print(f"  Loaded {len(shot_pool)} samples for shot pool")

    # Build subject index for subjective strategy
    subject_index = None
    if args.shot_strategy == 'subjective' and args.num_shots > 0:
        pool_s2subj, pool_subj2s = build_subject_index(shot_pool, args.pkl_dir, label="shot pool")
        val_s2subj, val_subj2s = build_subject_index(val_data, args.pkl_dir, label="val data")
        subject_index = (pool_s2subj, pool_subj2s, val_s2subj)

    shot_selector = ShotSelector(seed=args.seed, subject_index=subject_index)

    # =========================================================================
    # Center bias
    # =========================================================================
    default_center_bias = create_center_bias((args.resolution, args.resolution))
    use_data_centerbias = args.pkl_dir is not None and os.path.exists(args.pkl_dir)
    if use_data_centerbias:
        print(f"Using data-driven centerbias from: {args.pkl_dir}")
    else:
        print("Using synthetic Gaussian centerbias (sigma=20)")

    # Centerbias alpha
    centerbias_alpha = 0.0
    if args.centerbias_alpha is not None:
        centerbias_alpha = args.centerbias_alpha
        print(f"Using manual centerbias alpha: {centerbias_alpha}")
    elif args.tune_centerbias:
        if shot_pool:
            tune_pool = shot_pool
        else:
            tune_pool = val_data
        tune_rng = random.Random(args.seed)
        tune_samples = tune_rng.sample(tune_pool, min(args.tune_num_samples, len(tune_pool)))

        if args.metric_mode == 'grid':
            centerbias_alpha = tune_centerbias_alpha_grid(
                saliency_computer=saliency_computer,
                tune_samples=tune_samples,
                images_dir=args.images_dir,
                shot_pool=shot_pool,
                shot_selector=shot_selector,
                num_shots=args.num_shots,
                shot_strategy=args.shot_strategy,
                pkl_dir=args.pkl_dir,
                default_center_bias=default_center_bias,
                use_data_centerbias=use_data_centerbias,
                resolution=args.resolution,
                batch_size=args.batch_size,
            )
        else:
            print("Centerbias tuning is only supported in grid mode; "
                  "skipping tuning (centerbias_alpha=0.0).")
    else:
        print("Centerbias augmentation: disabled (use --tune-centerbias or --centerbias-alpha)")

    # =========================================================================
    # Verify-only mode
    # =========================================================================
    if args.verify_only:
        print("\nVerify-only mode: Testing prompt building...")

        for i in range(min(3, len(val_data))):
            sample = val_data[i]
            image_path = Path(args.images_dir) / sample['images'][0]
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue

            image = Image.open(image_path).convert('RGB')
            conv = sample['conversations']
            text_prompt = conv[0]['value'].replace("<image>", "").strip()

            selected_shots = shot_selector.select(
                shot_pool, args.num_shots, args.shot_strategy, test_sample=sample
            )
            shot_examples = prepare_shot_examples(selected_shots, args.images_dir)

            print(f"\nSample {i}: {sample['images'][0]}")
            print(f"  Prompt: {text_prompt[:100]}...")
            print(f"  Shots: {len(shot_examples)}")

            prompt, mm_data = saliency_computer.build_prompt(
                image, text_prompt, shot_examples, partial_response=""
            )
            print(f"  Prompt length: {len(prompt)} chars")

        return

    # =========================================================================
    # Main evaluation loop
    # =========================================================================
    import time as _time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_label = args.metric_mode
    results_file = output_dir / f"unified_{mode_label}_{args.num_shots}shot_{timestamp}.json"

    all_results = []
    if args.resume_json:
        resume_path = Path(args.resume_json)
        if resume_path.exists():
            with open(resume_path, 'r') as f:
                resume_data = json.load(f)
            all_results = resume_data.get('results', [])
            print(f"Resuming from {resume_path}: {len(all_results)} samples already completed")
            results_file = resume_path
        else:
            print(f"Warning: resume file not found: {resume_path}, starting fresh")

    def _save_results_json():
        with open(results_file, 'w') as f:
            json.dump({
                'config': vars(args),
                'centerbias_alpha': centerbias_alpha,
                'metric_mode': args.metric_mode,
                'num_samples': len(all_results),
                'results': all_results,
            }, f, indent=2, default=str)

    _save_results_json()
    print(f"Results file: {results_file}")

    # Group samples by image
    image_groups = defaultdict(list)
    for idx, sample in enumerate(val_data):
        image_groups[sample['images'][0]].append((idx, sample))
    print(f"Grouped {len(val_data)} samples into {len(image_groups)} image groups")

    completed_sample_indices = {r['idx'] for r in all_results}

    for image_path, group_samples in tqdm(image_groups.items(), desc="Evaluating images"):
        try:
            if all(idx in completed_sample_indices for idx, _ in group_samples):
                continue

            # Load image ONCE
            full_image_path = Path(args.images_dir) / image_path
            if not full_image_path.exists():
                print(f"Image not found: {full_image_path}")
                continue

            image = Image.open(full_image_path).convert('RGB')

            # Load centerbias ONCE
            if use_data_centerbias:
                center_bias, pkl_path = load_centerbias_from_pkl(
                    image_path, args.pkl_dir, args.resolution
                )
                cb_source = 'data' if center_bias is not None else 'synthetic'
                if center_bias is None:
                    center_bias = default_center_bias
                    pkl_path = None
            else:
                center_bias = default_center_bias
                cb_source = 'synthetic'
                pkl_path = None

            # Select shots ONCE
            selected_shots = shot_selector.select(
                shot_pool, args.num_shots, args.shot_strategy,
                test_sample=group_samples[0][1]
            )
            shot_examples = prepare_shot_examples(selected_shots, args.images_dir)

            # Collect valid scanpaths
            valid_entries = []
            for idx, sample in group_samples:
                conv = sample['conversations']
                text_prompt = conv[0]['value'].replace("<image>", "").strip()
                gt_scanpath_str = conv[1]['value']
                gt_fixations = parse_scanpath_reduced(gt_scanpath_str)

                # Parse temporal/duration data if available
                gt_temporal = None
                gt_durations = None
                if is_temporal:
                    gt_temporal = parse_scanpath_temporal(gt_scanpath_str)
                elif is_durations:
                    gt_durations = parse_scanpath_temporal(gt_scanpath_str)  # Same 3-tuple regex

                if len(gt_fixations) < 2:
                    print(f"Sample {idx}: Not enough fixations ({len(gt_fixations)})")
                    continue
                valid_entries.append((idx, sample, text_prompt, gt_fixations, gt_temporal, gt_durations))

            if not valid_entries:
                continue

            # Build result dicts
            group_results = []
            for idx, sample, text_prompt, gt_fixations, gt_temporal, gt_durations in valid_entries:
                result = {
                    'idx': idx,
                    'image': image_path,
                    'num_fixations': len(gt_fixations),
                    'gt_fixations': gt_fixations,
                    'num_shots': len(shot_examples),
                    'shot_images': [s['images'][0] for s in selected_shots] if selected_shots else [],
                    'centerbias_source': cb_source,
                }
                if is_temporal and gt_temporal:
                    result['is_temporal'] = True
                    result['gt_timestamps'] = [t for _, _, t in gt_temporal]
                if is_durations and gt_durations:
                    result['is_durations'] = True
                    result['gt_durations'] = [d for _, _, d in gt_durations]
                if pkl_path:
                    result['centerbias_pkl'] = pkl_path
                group_results.append(result)

            text_prompts = [tp for _, _, tp, _, _, _ in valid_entries]
            all_gt_fixations = [gf for _, _, _, gf, _, _ in valid_entries]
            total_transitions = sum(len(gf) - 1 for gf in all_gt_fixations)

            print(
                f"  Image {image_path}: {len(valid_entries)} scanpaths, "
                f"{total_transitions} transitions, cb={cb_source}, mode={args.metric_mode}",
                flush=True
            )

            _group_t0 = _time.time()

            # =================================================================
            # GRID MODE
            # =================================================================
            if args.metric_mode == 'grid':
                all_scanpath_grids = saliency_computer.compute_multi_scanpath_distributions(
                    image, text_prompts, shot_examples,
                    all_gt_fixations, args.resolution, args.batch_size
                )

                for (idx, sample, text_prompt, gt_fixations, gt_temporal, gt_durations), result, scanpath_grids in zip(
                    valid_entries, group_results, all_scanpath_grids
                ):
                    fixation_metrics = []
                    saved_grids = []
                    for i, log_density in enumerate(scanpath_grids):
                        target = gt_fixations[i + 1]

                        valid_mask = np.isfinite(log_density)
                        if not valid_mask.any():
                            continue
                        log_density = log_density - min(0.0, logsumexp(log_density[valid_mask]))

                        if centerbias_alpha > 0:
                            log_density = log_density + centerbias_alpha * center_bias
                            log_density = log_density - min(0.0, logsumexp(log_density))

                        if args.save_grids:
                            saved_grids.append(log_density.copy())

                        ig = compute_information_gain(log_density, center_bias, target, args.resolution)
                        auc = compute_auc(log_density, target, args.resolution)
                        nss = compute_nss(log_density, target, args.resolution)
                        log_nss = compute_log_nss(log_density, target, args.resolution)

                        target_x = min(args.resolution - 1, max(0, target[0]))
                        target_y = min(args.resolution - 1, max(0, target[1]))
                        ll = float(log_density[target_y, target_x])

                        fixation_metrics.append({
                            'idx': i + 1, 'target': target,
                            'ig': ig, 'auc': auc, 'nss': nss, 'log_nss': log_nss, 'll': ll,
                        })

                    if args.save_grids and saved_grids:
                        grids_dir = output_dir / "grids"
                        grids_dir.mkdir(exist_ok=True)
                        np.savez_compressed(
                            grids_dir / f"sample_{idx:05d}.npz",
                            grids=np.stack(saved_grids).astype(np.float16),
                            gt_fixations=np.array(gt_fixations, dtype=np.int16),
                            image=image_path,
                        )

                    if fixation_metrics:
                        result['lp_mean_ig'] = np.mean([m['ig'] for m in fixation_metrics])
                        result['lp_mean_auc'] = np.mean([m['auc'] for m in fixation_metrics])
                        result['lp_mean_nss'] = np.mean([m['nss'] for m in fixation_metrics])
                        if fixation_metrics[0].get('log_nss') is not None:
                            result['lp_mean_log_nss'] = np.mean([m['log_nss'] for m in fixation_metrics])
                        result['lp_mean_ll'] = np.mean([m['ll'] for m in fixation_metrics])
                        result['lp_fixation_metrics'] = fixation_metrics

                        _elapsed = _time.time() - _group_t0
                        print(
                            f"    Sample {idx} [grid]: "
                            f"IG={result['lp_mean_ig']:.2f}, "
                            f"AUC={result['lp_mean_auc']:.4f}, "
                            f"LL={result['lp_mean_ll']:.2f}  "
                            f"({_elapsed:.1f}s)",
                            flush=True
                        )

            # =================================================================
            # ASSUME_NORMALIZED MODE
            # =================================================================
            elif args.metric_mode == 'assume_normalized':
                xy_sep = saliency_computer._xy_separator or ", "

                for (idx, sample, text_prompt, gt_fixations, gt_temporal, gt_durations), result in zip(
                    valid_entries, group_results
                ):
                    gt_transition_info = []
                    gt_base_prompts = []
                    gt_mm_data = None

                    # Use temporal/duration fixations for formatting if available
                    fmt_fixations = gt_temporal if gt_temporal else (gt_durations if gt_durations else gt_fixations)

                    for i in range(1, len(gt_fixations)):
                        previous = fmt_fixations[:i]
                        target = gt_fixations[i]
                        partial_base = saliency_computer.format_partial_scanpath(
                            previous, xy_separator=xy_sep,
                            temporal=is_temporal,
                            durations=is_durations,
                        )
                        bp, md = saliency_computer.build_prompt(
                            image, text_prompt, shot_examples, partial_base
                        )
                        if gt_mm_data is None:
                            gt_mm_data = md
                        gt_base_prompts.append(bp)
                        tinfo = {
                            'scanpath_idx': 0,
                            'fixation_idx': i,
                            'gt_target': target,
                            'base_prompt': bp,
                        }
                        if gt_temporal:
                            tinfo['gt_timestamp'] = gt_temporal[i][2]
                        if gt_durations:
                            tinfo['gt_duration'] = gt_durations[i][2]
                        gt_transition_info.append(tinfo)

                    if not gt_transition_info:
                        result['lp_mean_ll'] = None
                        result['lp_fixation_lls'] = []
                        continue

                    # Detect separator if needed
                    if saliency_computer._xy_separator is None:
                        saliency_computer.detect_xy_separator(gt_base_prompts[0], gt_mm_data)
                        xy_sep = saliency_computer._xy_separator

                    # Score only GT coordinates (pass empty mc_coords)
                    per_t_lls = saliency_computer.score_coordinates(
                        gt_base_prompts, gt_mm_data, gt_transition_info,
                        [], xy_sep, args.batch_size,
                    )

                    lls = [
                        per_t_lls[t].get(gt_transition_info[t]['gt_target'], -20.0)
                        for t in range(len(gt_transition_info))
                    ]

                    log_z = args.log_z if args.log_z is not None else 0.0
                    igs = []
                    norm_lls = []
                    for t, tinfo in enumerate(gt_transition_info):
                        norm_ll = lls[t] - log_z
                        norm_lls.append(norm_ll)
                        x_gt, y_gt = tinfo['gt_target']
                        x_idx = min(args.resolution - 1, max(0, x_gt))
                        y_idx = min(args.resolution - 1, max(0, y_gt))
                        log_cb = center_bias[y_idx, x_idx]
                        ig = (norm_ll - log_cb) / np.log(2)
                        igs.append(ig)

                    result['lp_mean_ig'] = np.mean(igs) if igs else None
                    result['lp_fixation_igs'] = igs
                    result['lp_mean_ll'] = np.mean(norm_lls) if norm_lls else None
                    result['lp_fixation_lls'] = norm_lls
                    result['log_z'] = log_z

                    # Temporal scoring: score GT timestamp digits
                    if is_temporal and gt_temporal:
                        temporal_lls = saliency_computer.score_temporal_gt(
                            gt_base_prompts, gt_mm_data, gt_transition_info,
                            xy_sep, args.batch_size,
                        )
                        result['temporal_fixation_lls'] = temporal_lls
                        result['temporal_mean_ll'] = np.mean(temporal_lls) if temporal_lls else None

                    # Duration scoring: score GT duration digits + greedy prediction
                    if is_durations and gt_durations:
                        duration_lls = saliency_computer.score_duration_gt(
                            gt_base_prompts, gt_mm_data, gt_transition_info,
                            xy_sep, args.batch_size,
                        )
                        result['duration_fixation_lls'] = duration_lls
                        result['duration_mean_ll'] = np.mean(duration_lls) if duration_lls else None

                        # Greedy duration prediction + MSE
                        pred_durations = saliency_computer.predict_duration_greedy(
                            gt_base_prompts, gt_mm_data, gt_transition_info,
                            xy_sep, args.batch_size,
                        )
                        gt_durs = [tinfo['gt_duration'] for tinfo in gt_transition_info]
                        duration_errors = [(p - g) ** 2 for p, g in zip(pred_durations, gt_durs)]
                        result['duration_pred'] = pred_durations
                        result['duration_gt'] = gt_durs
                        result['duration_mse'] = np.mean(duration_errors) if duration_errors else None

                        # Debug: print GT vs pred when MSE is suspicious
                        if result['duration_mse'] is not None and (
                            result['duration_mse'] == 0 or result['duration_mse'] > 100000
                        ):
                            print(f"      DEBUG DMSE={result['duration_mse']:.0f}: "
                                  f"GT={gt_durs}, pred={pred_durations}", flush=True)

                    _elapsed = _time.time() - _group_t0
                    _z_label = f"logZ={args.log_z}" if args.log_z is not None else "Z~1"
                    _temporal_str = ""
                    if is_temporal and result.get('temporal_mean_ll') is not None:
                        _temporal_str = f", TLL={result['temporal_mean_ll']:.2f}"
                    if is_durations and result.get('duration_mean_ll') is not None:
                        _mse_str = f", DMSE={result['duration_mse']:.0f}" if result.get('duration_mse') is not None else ""
                        _temporal_str = f", DLL={result['duration_mean_ll']:.2f}{_mse_str}"
                    print(
                        f"    Sample {idx} [{_z_label}]: "
                        f"IG={result['lp_mean_ig']:.2f}, "
                        f"LL={result['lp_mean_ll']:.2f}"
                        f"{_temporal_str}  "
                        f"({_elapsed:.1f}s)",
                        flush=True
                    )

            # Save results for this image group
            all_results.extend(group_results)
            completed_sample_indices.update(r['idx'] for r in group_results)
            _save_results_json()

        except Exception as e:
            print(f"Error on image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # =========================================================================
    # Aggregate results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Metric mode: {args.metric_mode}")
    print(f"Num shots: {args.num_shots}")
    print(f"Shot strategy: {args.shot_strategy}")
    print(f"Centerbias alpha: {centerbias_alpha}")

    if all_results:
        lp_igs = [r['lp_mean_ig'] for r in all_results if r.get('lp_mean_ig') is not None]
        if lp_igs:
            print(f"\n--- {args.metric_mode.upper()} Metrics ---")
            print(f"  IG:  {np.mean(lp_igs):.3f} +/- {np.std(lp_igs):.3f}")
        lp_aucs = [r['lp_mean_auc'] for r in all_results if r.get('lp_mean_auc') is not None]
        if lp_aucs:
            print(f"  AUC: {np.mean(lp_aucs):.4f} +/- {np.std(lp_aucs):.4f}")
        lp_nsss = [r['lp_mean_nss'] for r in all_results if r.get('lp_mean_nss') is not None]
        if lp_nsss:
            print(f"  NSS: {np.mean(lp_nsss):.3f} +/- {np.std(lp_nsss):.3f}")
        lp_nss_corr = [r['lp_mean_nss_corrected'] for r in all_results if r.get('lp_mean_nss_corrected') is not None]
        if lp_nss_corr:
            print(f"  NSS (corrected): {np.mean(lp_nss_corr):.3f} +/- {np.std(lp_nss_corr):.3f}")
        lp_log_nsss = [r['lp_mean_log_nss'] for r in all_results if r.get('lp_mean_log_nss') is not None]
        if lp_log_nsss:
            print(f"  LogNSS: {np.mean(lp_log_nsss):.3f} +/- {np.std(lp_log_nsss):.3f}")
        lp_ms_nsss = [r['lp_mean_ms_nss'] for r in all_results if r.get('lp_mean_ms_nss') is not None]
        if lp_ms_nsss:
            print(f"  MS-NSS: {np.mean(lp_ms_nsss):.3f} +/- {np.std(lp_ms_nsss):.3f}")
        lp_lls = [r.get('lp_mean_ll') for r in all_results if r.get('lp_mean_ll') is not None]
        if lp_lls:
            print(f"  LL:  {np.mean(lp_lls):.3f} +/- {np.std(lp_lls):.3f}")
        temporal_lls = [r.get('temporal_mean_ll') for r in all_results if r.get('temporal_mean_ll') is not None]
        if temporal_lls:
            print(f"  Temporal LL: {np.mean(temporal_lls):.3f} +/- {np.std(temporal_lls):.3f}")
        duration_lls = [r.get('duration_mean_ll') for r in all_results if r.get('duration_mean_ll') is not None]
        if duration_lls:
            print(f"  Duration LL: {np.mean(duration_lls):.3f} +/- {np.std(duration_lls):.3f}")
        duration_mses = [r.get('duration_mse') for r in all_results if r.get('duration_mse') is not None]
        if duration_mses:
            print(f"  Duration MSE: {np.mean(duration_mses):.1f} +/- {np.std(duration_mses):.1f}")
            print(f"  Duration RMSE: {np.sqrt(np.mean(duration_mses)):.1f} ms")

    # =========================================================================
    # Visualization (grid mode only)
    # =========================================================================
    if args.metric_mode == 'grid' and not args.skip_viz and all_results:
        visualize_samples_by_ig(
            all_results, val_data, saliency_computer,
            shot_pool, shot_selector, args.num_shots, args.shot_strategy,
            args.images_dir, str(output_dir),
            num_samples=args.num_viz_samples,
            resolution=args.resolution,
        )

    _save_results_json()
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
