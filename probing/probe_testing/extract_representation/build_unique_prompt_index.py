#!/usr/bin/env python3
"""
Build deduplication index for all unique (profile_id, item_key) pairs.

Reads cells_main_prompts_tier2.json and cells_moderator_prompts_tier2.json,
takes the union of all (profile_id, item_key) pairs (deduplicating across both files),
and assigns each a stable integer row index.

Output: prompt_index.json
  {"profile_id__item_key": row_index, ...}
  1,584,000 entries, confirming the exact count:
  36,000 profiles × 44 items = 1,584,000 unique prompts
  (Includes all 44 items: 12 direct Tier-2 outcomes + 12 trust sub-items + 20 Tier-1 only)

Usage:
  python build_unique_prompt_index.py \\
    --cells_main ../prompt/cells_main_prompts_tier2.json \\
    --cells_moderator ../prompt/cells_moderator_prompts_tier2.json \\
    --output_file prompt_index.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("build_unique_prompt_index")


def setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(console)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells_main",
        default="../prompt/cells_main_prompts_tier2.json",
        help="Path to cells_main_prompts_tier2.json"
    )
    parser.add_argument(
        "--cells_moderator",
        default="../prompt/cells_moderator_prompts_tier2.json",
        help="Path to cells_moderator_prompts_tier2.json"
    )
    parser.add_argument(
        "--output_file",
        default="prompt_index.json",
        help="Output index file"
    )
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 78)
    log.info("BUILD UNIQUE PROMPT INDEX")
    log.info("=" * 78)
    log.info(f"Main cells file: {args.cells_main}")
    log.info(f"Moderator cells file: {args.cells_moderator}")

    # Load both cell files
    log.info("Loading cells_main_prompts_tier2.json...")
    with open(args.cells_main) as f:
        cells_main = json.load(f)
    log.info(f"Loaded {len(cells_main)} main cells")

    log.info("Loading cells_moderator_prompts_tier2.json...")
    with open(args.cells_moderator) as f:
        cells_moderator = json.load(f)
    log.info(f"Loaded {len(cells_moderator)} moderator cells")

    # Collect all unique (profile_id, item_key) pairs
    log.info("Extracting unique (profile_id, item_key) pairs...")
    unique_pairs = set()

    # From main cells
    for cell_key, prompts_in_cell in cells_main.items():
        for prompt_info in prompts_in_cell:
            profile_id = prompt_info["profile_id"]
            item_key = prompt_info["item_key"]
            unique_pairs.add((profile_id, item_key))

    log.info(f"After main cells: {len(unique_pairs)} unique pairs")

    # From moderator cells
    for cell_key, prompts_in_cell in cells_moderator.items():
        for prompt_info in prompts_in_cell:
            profile_id = prompt_info["profile_id"]
            item_key = prompt_info["item_key"]
            unique_pairs.add((profile_id, item_key))

    log.info(f"After moderator cells: {len(unique_pairs)} unique pairs (deduplicated)")

    # Build index: assign stable row indices
    # Sort by (profile_id, item_key) for consistency across runs
    sorted_pairs = sorted(unique_pairs)
    prompt_index = {f"{pid}__{ikey}": idx for idx, (pid, ikey) in enumerate(sorted_pairs)}

    log.info(f"Built index with {len(prompt_index)} entries")

    # Verify expected count
    expected_count = 36_000 * 44  # 36,000 profiles × 44 total items (32 Tier-2-related + 12 Tier-1 only)
    if len(prompt_index) == expected_count:
        log.info(f"✓ Index count matches expected: {expected_count:,}")
    else:
        log.warning(f"Index count {len(prompt_index):,} != expected {expected_count:,}")

    # Write output
    log.info(f"Writing index to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(prompt_index, f)

    log.info("-" * 78)
    log.info("SUMMARY")
    log.info("-" * 78)
    log.info(f"Total unique (profile_id, item_key) pairs: {len(prompt_index):,}")
    log.info(f"Output file: {args.output_file}")
    log.info(f"File size: {Path(args.output_file).stat().st_size / (1024**2):.1f} MB")

    log.info("\n" + "=" * 78)
    log.info("✓ INDEX BUILT SUCCESSFULLY")
    log.info("=" * 78)


if __name__ == "__main__":
    setup_logging()
    main()
