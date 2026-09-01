#!/usr/bin/env python3
"""
Build cell-grouped representations and compute trust subscales/multidimensional.

After extract_unique_representations.py has finished, this script:
  1. Loads the representation store (representations.dat memmap)
  2. For each cell in cells_main_prompts_tier2.json and cells_moderator_prompts_tier2.json:
     - Looks up each prompt's row index via prompt_index.json
     - Averages the representations for that cell
  3. Computes trust subscales from 3 sub-item means each:
     - trust_competence = mean(trust_competence_1, _2, _3)
     - trust_integrity = mean(trust_integrity_1, _2, _3)
     - trust_benevolence = mean(trust_benevolence_1, _2, _3)
     - trust_openness = mean(trust_openness_1, _2, _3)
  4. Computes trust_multidimensional = mean(subscales)
  5. Writes final output files (ONE write per file, at the end)

This is O(n) in the number of cells (cheap!), no model calls, no redundant I/O.

Output:
  - group_representations_main.json (221 cells = 17 conditions × 13 outcomes)
  - group_representations_moderator.json (5,967 cells = 17 × 27 × 13 outcomes)

Usage:
  python build_cell_groups_from_store.py \\
    --representations_file representations.dat \\
    --prompt_index prompt_index.json \\
    --cells_main ../prompt/cells_main_prompts_tier2.json \\
    --cells_moderator ../prompt/cells_moderator_prompts_tier2.json \\
    --output_dir .
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

log = logging.getLogger("build_cell_groups_from_store")


def setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(console)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations_file", default="representations.dat",
                       help="Path to representations.dat (output from extract_unique_representations.py)")
    parser.add_argument("--prompt_index", default="prompt_index.json",
                       help="Path to prompt_index.json (output from build_unique_prompt_index.py)")
    parser.add_argument("--cells_main", default="../prompt/cells_main_prompts_tier2.json")
    parser.add_argument("--cells_moderator", default="../prompt/cells_moderator_prompts_tier2.json")
    parser.add_argument("--output_dir", default=".",
                       help="Output directory for group_representations_*.json files")
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 78)
    log.info("BUILD CELL GROUPS FROM REPRESENTATION STORE")
    log.info("=" * 78)
    log.info(f"Representations file: {args.representations_file}")
    log.info(f"Prompt index: {args.prompt_index}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load representations memmap
    log.info("Loading representations memmap...")
    representations = np.memmap(args.representations_file, dtype=np.float16, mode='r', shape=(1584000, 5120))
    log.info(f"Loaded memmap: shape={representations.shape}, dtype={representations.dtype}")

    # Load prompt index
    log.info("Loading prompt index...")
    with open(args.prompt_index) as f:
        prompt_index = json.load(f)
    log.info(f"Loaded {len(prompt_index):,} prompts indexed")

    # Load cell groupings
    log.info("Loading cell groupings...")
    with open(args.cells_main) as f:
        cells_main = json.load(f)
    log.info(f"Loaded {len(cells_main)} main cells")

    with open(args.cells_moderator) as f:
        cells_moderator = json.load(f)
    log.info(f"Loaded {len(cells_moderator)} moderator cells")

    # Build main cell groups
    log.info("Building main cell group representations...")
    group_representations_main = {}

    for cell_idx, (cell_key, prompts_in_cell) in enumerate(sorted(cells_main.items())):
        if (cell_idx + 1) % 50 == 0:
            log.info(f"  Main cell {cell_idx + 1}/{len(cells_main)}: {cell_key}")

        reps_for_cell = []
        for prompt_info in prompts_in_cell:
            profile_id = prompt_info["profile_id"]
            item_key = prompt_info["item_key"]
            key = f"{profile_id}__{item_key}"

            if key not in prompt_index:
                log.warning(f"Prompt {key} not in index, skipping")
                continue

            row_idx = prompt_index[key]
            rep = representations[row_idx].astype(np.float32)  # Convert float16 back to float32 for averaging
            reps_for_cell.append(rep)

        if reps_for_cell:
            avg_rep = np.mean(reps_for_cell, axis=0).tolist()
            group_representations_main[cell_key] = avg_rep
        else:
            log.warning(f"No representations found for main cell {cell_key}")

    log.info(f"Built {len(group_representations_main)} main cell groups")

    # Compute trust subscales and trust_multidimensional from trust sub-item representations
    log.info("Computing trust subscales and trust_multidimensional from sub-item representations...")
    trust_subscales_main = {}  # {condition: {subscale: rep_vector}}

    # Extract all unique conditions
    conditions = set()
    for cell_key in cells_main.keys():
        condition = cell_key.split("__")[0]
        conditions.add(condition)

    for condition in conditions:
        trust_subscales_main[condition] = {}

        # Compute each trust subscale from its 3 sub-items
        for subscale_name, sub_items in [
            ("trust_competence", ["trust_competence_1", "trust_competence_2", "trust_competence_3"]),
            ("trust_integrity", ["trust_integrity_1", "trust_integrity_2", "trust_integrity_3"]),
            ("trust_benevolence", ["trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3"]),
            ("trust_openness", ["trust_openness_1", "trust_openness_2", "trust_openness_3"]),
        ]:
            # Look up the 3 sub-item cell representations
            sub_item_reps = []
            for sub_item in sub_items:
                cell_key = f"{condition}__{sub_item}"
                if cell_key in group_representations_main:
                    sub_item_reps.append(np.array(group_representations_main[cell_key]))
                else:
                    log.warning(f"Missing cell representation for {cell_key}")

            if sub_item_reps:
                # Compute subscale as mean of 3 sub-items
                subscale_rep = np.mean(sub_item_reps, axis=0)
                trust_subscales_main[condition][subscale_name] = subscale_rep

    # Compute trust_multidimensional for each condition from 4 subscales
    for condition in conditions:
        subscale_reps = []
        for subscale_name in ["trust_competence", "trust_integrity", "trust_benevolence", "trust_openness"]:
            if subscale_name in trust_subscales_main[condition]:
                subscale_reps.append(trust_subscales_main[condition][subscale_name])

        if subscale_reps:
            tm_rep = np.mean(subscale_reps, axis=0)
            cell_key = f"{condition}__trust_multidimensional"
            group_representations_main[cell_key] = tm_rep.tolist()
            log.info(f"Computed trust_multidimensional for {condition}")
        else:
            log.warning(f"Could not compute trust_multidimensional for {condition}: missing subscales")

    log.info(f"Final main cells (with trust_multidimensional): {len(group_representations_main)}")

    # Filter out trust sub-items; keep only direct outcomes + computed multidimensional
    log.info("Filtering main cells to keep only direct outcomes + trust_multidimensional...")
    direct_outcomes = {
        "behavior_mean", "belief_post", "concern_mean", "distrust_post",
        "donation_ams", "funding_perceptions", "inst_trust_mean", "newsletter_signup",
        "policy_general", "policy_role_mean", "policy_specific_mean", "trust_post",
        "trust_multidimensional"
    }
    group_representations_main_filtered = {}
    for cell_key, rep_vector in group_representations_main.items():
        outcome = cell_key.split("__")[-1]  # Get last part after split
        if outcome in direct_outcomes:
            group_representations_main_filtered[cell_key] = rep_vector
    group_representations_main = group_representations_main_filtered
    log.info(f"After filtering: {len(group_representations_main)} main cells (expected 221)")

    # Build moderator cell groups
    log.info("Building moderator cell group representations...")
    group_representations_moderator = {}

    for cell_idx, (cell_key, prompts_in_cell) in enumerate(sorted(cells_moderator.items())):
        if (cell_idx + 1) % 500 == 0:
            log.info(f"  Moderator cell {cell_idx + 1}/{len(cells_moderator)}: {cell_key}")

        reps_for_cell = []
        for prompt_info in prompts_in_cell:
            profile_id = prompt_info["profile_id"]
            item_key = prompt_info["item_key"]
            key = f"{profile_id}__{item_key}"

            if key not in prompt_index:
                log.warning(f"Prompt {key} not in index, skipping")
                continue

            row_idx = prompt_index[key]
            rep = representations[row_idx].astype(np.float32)  # Convert float16 back to float32
            reps_for_cell.append(rep)

        if reps_for_cell:
            avg_rep = np.mean(reps_for_cell, axis=0).tolist()
            group_representations_moderator[cell_key] = avg_rep
        else:
            log.warning(f"No representations found for moderator cell {cell_key}")

    log.info(f"Built {len(group_representations_moderator)} moderator cell groups")

    # Compute trust subscales and trust_multidimensional for moderator cells
    log.info("Computing trust subscales and trust_multidimensional for moderator cells...")
    trust_subscales_moderator = {}  # {(condition, moderator, level): {subscale: rep_vector}}

    # Extract all unique (condition, moderator, level) combinations
    moderator_keys = set()
    for cell_key in cells_moderator.keys():
        parts = cell_key.split("__")
        if len(parts) == 4:  # condition__moderator__level__outcome
            condition, moderator, level = parts[0], parts[1], parts[2]
            moderator_keys.add((condition, moderator, level))

    for condition, moderator, level in moderator_keys:
        key = (condition, moderator, level)
        trust_subscales_moderator[key] = {}

        # Compute each trust subscale from its 3 sub-items
        for subscale_name, sub_items in [
            ("trust_competence", ["trust_competence_1", "trust_competence_2", "trust_competence_3"]),
            ("trust_integrity", ["trust_integrity_1", "trust_integrity_2", "trust_integrity_3"]),
            ("trust_benevolence", ["trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3"]),
            ("trust_openness", ["trust_openness_1", "trust_openness_2", "trust_openness_3"]),
        ]:
            # Look up the 3 sub-item cell representations
            sub_item_reps = []
            for sub_item in sub_items:
                cell_key = f"{condition}__{moderator}__{level}__{sub_item}"
                if cell_key in group_representations_moderator:
                    sub_item_reps.append(np.array(group_representations_moderator[cell_key]))
                # else: log warning is optional; missing cells should be rare

            if sub_item_reps:
                # Compute subscale as mean of 3 sub-items
                subscale_rep = np.mean(sub_item_reps, axis=0)
                trust_subscales_moderator[key][subscale_name] = subscale_rep

    # Compute trust_multidimensional for each (condition, moderator, level)
    for condition, moderator, level in moderator_keys:
        key = (condition, moderator, level)
        subscale_reps = []
        for subscale_name in ["trust_competence", "trust_integrity", "trust_benevolence", "trust_openness"]:
            if subscale_name in trust_subscales_moderator[key]:
                subscale_reps.append(trust_subscales_moderator[key][subscale_name])

        if subscale_reps:
            tm_rep = np.mean(subscale_reps, axis=0)
            cell_key = f"{condition}__{moderator}__{level}__trust_multidimensional"
            group_representations_moderator[cell_key] = tm_rep.tolist()
        # else: log if desired

    log.info(f"Final moderator cells (with trust_multidimensional): {len(group_representations_moderator)}")

    # Filter out trust sub-items; keep only direct outcomes + computed multidimensional
    log.info("Filtering moderator cells to keep only direct outcomes + trust_multidimensional...")
    group_representations_moderator_filtered = {}
    for cell_key, rep_vector in group_representations_moderator.items():
        outcome = cell_key.split("__")[-1]  # Get last part after split
        if outcome in direct_outcomes:
            group_representations_moderator_filtered[cell_key] = rep_vector
    group_representations_moderator = group_representations_moderator_filtered
    log.info(f"After filtering: {len(group_representations_moderator)} moderator cells (expected 5,967)")

    # Write output files (single write each, at the end)
    log.info("Writing output files...")
    main_output_file = output_dir / "group_representations_main.json"
    with open(main_output_file, "w") as f:
        json.dump(group_representations_main, f, indent=2)
    log.info(f"Wrote {main_output_file}")

    moderator_output_file = output_dir / "group_representations_moderator.json"
    with open(moderator_output_file, "w") as f:
        json.dump(group_representations_moderator, f, indent=2)
    log.info(f"Wrote {moderator_output_file}")

    # Summary
    log.info("-" * 78)
    log.info("SUMMARY")
    log.info("-" * 78)
    log.info(f"Main cell groups (including trust_multidimensional): {len(group_representations_main)} (expected 221)")
    log.info(f"  = 17 conditions × 13 outcomes (12 direct + computed trust_multidimensional)")
    log.info(f"Moderator cell groups (including trust_multidimensional): {len(group_representations_moderator)} (expected 5,967)")
    log.info(f"  = 17 conditions × 27 moderator-levels × 13 outcomes")
    log.info(f"Main output: {main_output_file}")
    log.info(f"Moderator output: {moderator_output_file}")

    # Verification
    expected_main = 221  # 17 × 13 (12 direct + computed trust_multidimensional)
    expected_moderator = 5967  # 17 × 27 × 13
    if len(group_representations_main) == expected_main and len(group_representations_moderator) == expected_moderator:
        log.info(f"✓ Cell counts match expected values")
    else:
        log.warning(f"✗ Cell counts mismatch:")
        log.warning(f"  Main: {len(group_representations_main)} != {expected_main}")
        log.warning(f"  Moderator: {len(group_representations_moderator)} != {expected_moderator}")

    log.info("\n" + "=" * 78)
    log.info("✓ CELL GROUPS BUILT SUCCESSFULLY")
    log.info("=" * 78)


if __name__ == "__main__":
    setup_logging()
    main()
