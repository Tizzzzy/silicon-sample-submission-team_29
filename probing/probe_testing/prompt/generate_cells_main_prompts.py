#!/usr/bin/env python3
"""
Group all 1,584,000 prompts by (condition × outcome) cells for Tier-2 main file.

This script reads prompts.jsonl and items_meta.json, then groups prompts into
cells matching the Tier-2 main file structure:
  - One cell per (condition, submission_var/outcome)
  - 17 conditions × 13 outcomes = 221 cells

Output: cells_main_prompts.json
  Key format: "{condition}__{outcome}"
  Value: list of [profile_id, item_key, submission_var, context_snippet]

This tells you which prompts belong to which aggregate cell, useful for
post-inference aggregation.

Usage:
  python generate_cells_main_prompts.py [--output_dir .]
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("generate_cells_main_prompts")


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
    log.info("GROUP PROMPTS BY TIER-2 MAIN CELLS (condition × outcome)")
    log.info("=" * 78)

    # Load items metadata
    log.info("Loading items_meta.json...")
    with open(output_dir / "items_meta.json") as f:
        items_meta = json.load(f)
    log.info(f"Loaded {len(items_meta)} items")

    # Group prompts by (condition, outcome)
    log.info("Grouping prompts by (condition × outcome)...")
    cells = defaultdict(list)
    n_prompts = 0

    with open(output_dir / "prompts.jsonl") as f:
        for line in f:
            profile = json.loads(line)
            profile_id = profile["profile_id"]
            condition = profile["condition"]
            context = profile["context"]

            # For each item, add to the appropriate cell
            for item_meta in items_meta:
                item_key = item_meta["item_key"]
                submission_var = item_meta["submission_var"]

                # Cell key: condition__outcome
                cell_key = f"{condition}__{submission_var}"

                cells[cell_key].append({
                    "profile_id": profile_id,
                    "item_key": item_key,
                    "submission_var": submission_var,
                })

                n_prompts += 1

    log.info(f"Grouped {n_prompts:,} prompts into {len(cells)} cells")

    # Write output
    log.info("Writing output...")
    output_path = output_dir / "cells_main_prompts.json"
    with open(output_path, "w") as f:
        json.dump(cells, f, indent=2)
    log.info(f"Wrote {output_path}")

    # Summary statistics
    log.info("-" * 78)
    log.info("SUMMARY STATISTICS")
    log.info("-" * 78)
    log.info(f"Total cells: {len(cells)}")
    log.info(f"Total prompts grouped: {n_prompts:,}")

    # Check structure
    log.info("\nExpected structure:")
    log.info(f"  Conditions: 17")
    log.info(f"  Outcomes: 13")
    log.info(f"  Expected cells: 17 × 13 = 221")
    log.info(f"  Actual cells: {len(cells)}")

    if len(cells) == 221:
        log.info("  ✓ Cell count matches expected")
    else:
        log.warning(f"  ✗ Expected 221 cells, got {len(cells)}")

    # Sample output
    log.info("\nSample cells (first 3):")
    for i, (key, prompts) in enumerate(sorted(cells.items())[:3]):
        log.info(f"  {key}: {len(prompts)} prompts")
        if prompts:
            log.info(f"    Example: {prompts[0]}")

    log.info("\n" + "=" * 78)
    log.info("✓ CELLS MAIN PROMPTS GENERATED")
    log.info("=" * 78)


if __name__ == "__main__":
    setup_logging()
    main()
