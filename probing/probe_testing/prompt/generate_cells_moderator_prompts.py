#!/usr/bin/env python3
"""
Group all 1,584,000 prompts by (condition × moderator × level × outcome) cells for Tier-2 moderator file.

This script reads prompts.jsonl and items_meta.json, then groups prompts into
cells matching the Tier-2 moderator file structure:
  - One cell per (condition, moderator, moderator_level, submission_var)
  - 17 conditions × 6 moderators × ~27 levels × 13 outcomes = ~5,967 cells

Moderators (6):
  - gender (3 levels: Male, Female, Other)
  - age_band (4 levels: 18-29, 30-44, 45-59, 60+)
  - race (5 levels: White/Caucasian, Black/African American, Hispanic/Latino, Asian/Asian American, Other)
  - education (6 levels: <HS, HS/GED, Some college, Bachelor's, Master's/Professional, Doctorate)
  - income (5 levels: <$30k, $30k-$55.9k, $56k-$99.9k, $100k-$167.9k, $168k+)
  - party (4 levels: Republican, Democrat, Independent, Other)

Output: cells_moderator_prompts.json
  Key format: "{condition}__{moderator}__{level}__{outcome}"
  Value: list of [profile_id, item_key, submission_var]

Usage:
  python generate_cells_moderator_prompts.py [--output_dir .]
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("generate_cells_moderator_prompts")

# The 6 moderators that go into the file
MODERATORS = ["gender", "age_band", "race", "education", "income", "party"]


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
        "--output_dir",
        default=".",
        help="Output directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 78)
    log.info("GROUP PROMPTS BY TIER-2 MODERATOR CELLS")
    log.info("(condition × moderator × level × outcome)")
    log.info("=" * 78)

    # Load items metadata
    log.info("Loading items_meta.json...")
    with open(output_dir / "items_meta.json") as f:
        items_meta = json.load(f)
    log.info(f"Loaded {len(items_meta)} items")

    # Group prompts by (condition, moderator, level, outcome)
    log.info("Grouping prompts by (condition × moderator × level × outcome)...")
    cells = defaultdict(list)
    moderator_levels = defaultdict(set)  # Track unique levels per moderator
    n_prompts = 0

    with open(output_dir / "prompts.jsonl") as f:
        for line in f:
            profile = json.loads(line)
            profile_id = profile["profile_id"]
            condition = profile["condition"]

            # For each moderator dimension
            for moderator in MODERATORS:
                moderator_level = profile[moderator]
                moderator_levels[moderator].add(moderator_level)

                # For each item, add to the appropriate cell
                for item_meta in items_meta:
                    item_key = item_meta["item_key"]
                    submission_var = item_meta["submission_var"]

                    # Cell key: condition__moderator__level__outcome
                    cell_key = f"{condition}__{moderator}__{moderator_level}__{submission_var}"

                    cells[cell_key].append({
                        "profile_id": profile_id,
                        "item_key": item_key,
                        "submission_var": submission_var,
                    })

                    n_prompts += 1

    log.info(f"Grouped {n_prompts:,} prompts into {len(cells)} cells")

    # Write output
    log.info("Writing output...")
    output_path = output_dir / "cells_moderator_prompts.json"
    with open(output_path, "w") as f:
        json.dump(cells, f, indent=2)
    log.info(f"Wrote {output_path}")

    # Summary statistics
    log.info("-" * 78)
    log.info("SUMMARY STATISTICS")
    log.info("-" * 78)
    log.info(f"Total cells: {len(cells)}")
    log.info(f"Total prompts grouped: {n_prompts:,}")

    # Moderator level breakdown
    log.info("\nModerator levels (unique values per moderator):")
    total_levels = 0
    for moderator in MODERATORS:
        levels = sorted(moderator_levels[moderator])
        n_levels = len(levels)
        total_levels += n_levels
        log.info(f"  {moderator}: {n_levels} levels")
        log.info(f"    {levels}")

    log.info(f"\nTotal unique level combinations: {total_levels}")

    # Check structure
    log.info("\nExpected structure:")
    log.info(f"  Conditions: 17")
    log.info(f"  Moderators: 6")
    log.info(f"  Total moderator-levels: {total_levels}")
    log.info(f"  Outcomes: 13")
    expected = 17 * total_levels * 13
    log.info(f"  Expected cells: 17 × {total_levels} × 13 = {expected}")
    log.info(f"  Actual cells: {len(cells)}")

    if len(cells) == expected:
        log.info("  ✓ Cell count matches expected")
    else:
        log.warning(f"  ✗ Expected {expected} cells, got {len(cells)}")

    # Sample output
    log.info("\nSample cells (first 5):")
    for i, (key, prompts) in enumerate(sorted(cells.items())[:5]):
        log.info(f"  {key}: {len(prompts)} prompts")

    log.info("\n" + "=" * 78)
    log.info("✓ CELLS MODERATOR PROMPTS GENERATED")
    log.info("=" * 78)


if __name__ == "__main__":
    setup_logging()
    main()
