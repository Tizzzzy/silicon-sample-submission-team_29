#!/usr/bin/env python3
"""
Generate all individual-level Tier-1 prompts for the benchmark.

This script produces the 1,584,000 individual-level prompts (36,000 profiles × 44 items)
needed for a Tier-1 submission, without running inference. Prompts are stored compactly:

- items_meta.json: 44 entries (all item metadata, item-independent)
- prompts.jsonl: 36,000 lines (one per profile, context shared across all 44 items)
- assemble.py: helper to reconstruct any full prompt from the compact storage
- manifest.json: run metadata

All prompts are built using the exact same logic as LLM_simulation/generate_outcomes.py:
demographics, stimulus, item questions, response formats, and constraints (regex/max_tokens).

Usage:
python generate_prompts.py \\
--profile_pool ../../synthetic_profiles/profiles_pool.csv \\
--questionnaire ../../survey/questionnaire.txt \\
--style bio \\
--seed 2026 \\
--output_dir . \\
    [--max_profiles 20]  # optional, for testing
    python generate_prompts.py --profile_pool /projects/p32143/silicon-sample-submission/synthetic_profiles/profiles_pool.csv --questionnaire /projects/p32143/silicon-sample-submission/survey/questionnaire.txt --style bio --seed 2026 --output_dir .
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add LLM_simulation to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "LLM_simulation"))

from items import ITEMS, BY_KEY, load_newsletter_offer
from prompt_qa import demographic_preamble, response_block, build_item_prompt, ItemPrompt, _TRANSITION, _FINAL_TRANSITION
from stimuli import parse_stimuli, stimulus_for, render_extreme_weather

log = logging.getLogger("generate_prompts")


def setup_logging(log_file: Optional[Path] = None) -> None:
    """Configure logging to console and optionally to file."""
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(console)
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"))
        log.addHandler(handler)


def respondent_stimulus(profile: Dict, stimuli: Dict[str, str], rng: random.Random) -> str:
    """The exact text this respondent sees, including the two adaptive arms.

    Copied verbatim from generate_outcomes.py:85 for fidelity.
    """
    condition = profile["condition"]
    if condition == "Extreme weather predictions":
        return render_extreme_weather(stimuli[condition], profile.get("state"))
    return stimulus_for(condition, stimuli, rng)


def build_items_meta(newsletter_offer: str) -> List[Dict]:
    """Build compact metadata for all 44 items.

    Each entry contains: item_key, submission_var, reverse, scale, regex, max_tokens,
    question_block, and extra_context (non-null only for newsletter).

    Returns list of 44 dicts, in the same order as ITEMS.
    """
    items_meta = []
    for item in ITEMS:
        scale_line, cue, regex = response_block(item)

        # Build the item's question block (suffix), exactly as build_item_prompt does
        q = []
        if item.intro:
            q.append(item.intro)
        q.append(item.text)
        if scale_line:
            q.append(scale_line)
        q.append(cue)
        question_block = "\n\n".join(q)

        # Newsletter item needs the offer page as extra context
        extra_context = None
        if item.key == "newsletter":
            extra_context = newsletter_offer

        meta = {
            "item_key": item.key,
            "submission_var": item.submission_var,
            "reverse": item.reverse,
            "scale": item.scale,
            "regex": regex,
            "max_tokens": 4 if item.scale != "binary" else 2,
            "question_block": question_block,
            "extra_context": extra_context,
        }
        items_meta.append(meta)

    return items_meta


def assemble_prompt(context: str, item_meta: Dict, extra_context_override: Optional[str] = None) -> str:
    """Reconstruct a full prompt from compact storage.

    This must be mathematically identical to build_item_prompt(...).text.

    Args:
        context: demographic preamble + transition + stimulus + final transition
        item_meta: entry from items_meta list
        extra_context_override: override the item_meta's extra_context (for testing)

    Returns:
        Full prompt string, ready to send to LLM.
    """
    parts = [context]

    extra = extra_context_override if extra_context_override is not None else item_meta["extra_context"]
    if extra:
        parts.append(extra)

    parts.append(item_meta["question_block"])

    return "\n\n".join(parts)


def fidelity_self_test(
    profiles: pd.DataFrame,
    stimuli: Dict[str, str],
    newsletter_offer: str,
    style: str,
    seed: int,
    items_meta: List[Dict],
) -> None:
    """Verify that assemble_prompt matches build_item_prompt exactly.

    Samples a handful of profiles across different conditions and checks all 44
    items for exact string match. Aborts loudly on failure.
    """
    log.info("Running fidelity self-test...")

    # Pick a few diverse profiles: first, last, one control, one "Extreme weather predictions"
    test_indices = [0, len(profiles) - 1]
    for idx, row in profiles.iterrows():
        if row["condition"] == "control":
            test_indices.append(idx)
        if row["condition"] == "Extreme weather predictions":
            test_indices.append(idx)
    test_indices = sorted(set(test_indices))[:5]  # limit to 5 to keep test fast

    rng = random.Random(seed)
    n_checked = 0

    for test_idx in test_indices:
        profile_row = profiles.iloc[test_idx]
        profile_dict = profile_row.to_dict()
        pid = str(profile_dict["profile_id"])

        # Recompute context as the script would
        stim = respondent_stimulus(profile_dict, stimuli, rng)
        preamble = demographic_preamble(profile_dict, style)
        context = "\n\n".join([preamble, _TRANSITION, stim, _FINAL_TRANSITION])

        for item_meta_entry in items_meta:
            item = BY_KEY[item_meta_entry["item_key"]]
            extra = item_meta_entry["extra_context"]

            # Build via build_item_prompt (reference)
            ref_prompt = build_item_prompt(
                profile_dict, stim, item, style=style, extra_context=extra
            ).text

            # Build via assemble_prompt (our method)
            our_prompt = assemble_prompt(context, item_meta_entry, extra_context_override=extra)

            if ref_prompt != our_prompt:
                log.error(f"FIDELITY TEST FAILED for {pid}/{item_meta_entry['item_key']}")
                log.error(f"Reference:\n{ref_prompt[:200]}...")
                log.error(f"Our version:\n{our_prompt[:200]}...")
                raise AssertionError("assemble_prompt does not match build_item_prompt")

            n_checked += 1

    log.info(f"✓ Fidelity test passed ({n_checked} prompts checked)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile_pool",
        default=str(Path(__file__).resolve().parents[3] / "synthetic_profiles" / "profiles_pool.csv"),
        help="Path to profile pool CSV"
    )
    parser.add_argument(
        "--questionnaire",
        default=str(Path(__file__).resolve().parents[3] / "survey" / "questionnaire.txt"),
        help="Path to questionnaire.txt"
    )
    parser.add_argument(
        "--style",
        default="bio",
        choices=["qa", "bio", "portray"],
        help="Demographic conditioning style"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Output directory"
    )
    parser.add_argument(
        "--max_profiles",
        type=int,
        default=None,
        help="Limit to first N profiles (for testing)"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / f"generate_prompts_{timestamp}.log")

    log.info("=" * 78)
    log.info("GENERATE ALL INDIVIDUAL-LEVEL TIER-1 PROMPTS")
    log.info("=" * 78)
    log.info(f"Profile pool: {args.profile_pool}")
    log.info(f"Questionnaire: {args.questionnaire}")
    log.info(f"Style: {args.style}")
    log.info(f"Seed: {args.seed}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Max profiles: {args.max_profiles or 'all'}")

    # Load data
    log.info("Loading profile pool...")
    profiles = pd.read_csv(args.profile_pool)
    if args.max_profiles:
        profiles = profiles.head(args.max_profiles)
    log.info(f"Loaded {len(profiles):,} profiles")

    log.info("Parsing stimuli...")
    stimuli = parse_stimuli(args.questionnaire)
    log.info(f"Parsed {len(stimuli)} stimulus blocks")

    log.info("Loading newsletter offer...")
    newsletter_offer = load_newsletter_offer(args.questionnaire)

    # Build item metadata once
    log.info("Building items metadata...")
    items_meta = build_items_meta(newsletter_offer)
    log.info(f"Built metadata for {len(items_meta)} items")

    # Fidelity self-test
    fidelity_self_test(profiles, stimuli, newsletter_offer, args.style, args.seed, items_meta)

    # Main loop: generate prompts
    log.info(f"Generating {len(profiles):,} × {len(ITEMS)} = {len(profiles) * len(ITEMS):,} prompts...")

    rng = random.Random(args.seed)
    prompts_path = output_dir / "prompts.jsonl"
    n_written = 0

    with open(prompts_path, "w") as f_out:
        for idx, (_, profile_row) in enumerate(profiles.iterrows()):
            profile_dict = profile_row.to_dict()
            pid = str(profile_dict["profile_id"])
            condition = profile_dict["condition"]

            # Compute context (shared across all 44 items for this profile)
            stim = respondent_stimulus(profile_dict, stimuli, rng)
            preamble = demographic_preamble(profile_dict, args.style)
            context = "\n\n".join([preamble, _TRANSITION, stim, _FINAL_TRANSITION])

            # Write one line per profile
            record = {
                "profile_id": pid,
                "condition": condition,
                "gender": profile_dict["gender"],
                "age_band": profile_dict["age_band"],
                "race": profile_dict["race"],
                "education": profile_dict["education"],
                "income": profile_dict["income"],
                "party": profile_dict["party"],
                "state": profile_dict.get("state"),
                "context": context,
            }
            f_out.write(json.dumps(record) + "\n")
            n_written += 1

            if (idx + 1) % 5000 == 0:
                log.info(f"  Wrote {idx + 1:,} profiles")

    log.info(f"Wrote {n_written:,} profiles to {prompts_path}")

    # Write items metadata
    items_meta_path = output_dir / "items_meta.json"
    with open(items_meta_path, "w") as f:
        json.dump(items_meta, f, indent=2)
    log.info(f"Wrote items metadata to {items_meta_path}")

    # Write manifest
    manifest = {
        "timestamp": timestamp,
        "seed": args.seed,
        "style": args.style,
        "profile_pool": args.profile_pool,
        "questionnaire": args.questionnaire,
        "n_profiles": len(profiles),
        "n_items": len(ITEMS),
        "n_prompts_total": len(profiles) * len(ITEMS),
        "output_dir": str(output_dir),
        "note": "No inference has been run yet; these are prompts ready for LLM consumption.",
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Wrote manifest to {manifest_path}")

    # Summary statistics
    log.info("-" * 78)
    log.info("GENERATION COMPLETE")
    log.info("-" * 78)
    log.info(f"Profiles: {len(profiles):,}")
    log.info(f"Items per profile: {len(ITEMS)}")
    log.info(f"Total prompts: {len(profiles) * len(ITEMS):,}")
    log.info(f"Output files:")
    log.info(f"  prompts.jsonl: {prompts_path.stat().st_size / (1024**2):.1f} MB")
    log.info(f"  items_meta.json: {items_meta_path.stat().st_size / 1024:.1f} KB")
    log.info(f"  manifest.json: {manifest_path.stat().st_size / 1024:.1f} KB")
    log.info(f"  Total: {(prompts_path.stat().st_size + items_meta_path.stat().st_size + manifest_path.stat().st_size) / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()
