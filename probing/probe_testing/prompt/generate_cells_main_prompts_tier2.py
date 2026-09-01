#!/usr/bin/env python3
"""
Group prompts by outcomes for Tier-2 submission (main cells).

This script reads prompts.jsonl and items_meta.json, then groups prompts by
the 24 outcomes that need to be EXTRACTED for Tier-2:

12 direct Tier-2 outcomes (with survey items):
  trust_post, distrust_post, funding_perceptions, policy_role_mean,
  inst_trust_mean, belief_post, concern_mean, policy_general,
  policy_specific_mean, behavior_mean, donation_ams, newsletter_signup

12 trust sub-items (used to compute trust subscales):
  trust_competence_1, trust_competence_2, trust_competence_3
  trust_integrity_1, trust_integrity_2, trust_integrity_3
  trust_benevolence_1, trust_benevolence_2, trust_benevolence_3
  trust_openness_1, trust_openness_2, trust_openness_3

The 13th official Tier-2 outcome (trust_multidimensional) has NO items;
it is COMPUTED from the 4 trust subscale means in build_cell_groups_from_store.py.

Output: cells_main_prompts_tier2.json
  Key format: "{condition}__{outcome}"
  Value: list of [profile_id, item_key] that contribute to this outcome
  Total cells: 17 conditions × 24 outcomes = 408 cells

After extraction and representation aggregation, you'll compute:
  - 4 trust subscale means (from 3 sub-items each)
  - trust_multidimensional (from 4 subscale means)
  → Final output has 17 conditions × 13 outcomes = 221 rows + header = 222

Usage:
  python generate_cells_main_prompts_tier2.py [--output_dir .]
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("generate_cells_main_prompts_tier2")

# The 13 official Tier-2 outcomes for benchmark scoring
# BUT: trust_multidimensional has NO items (it's computed from 4 subscales)
# So we extract 12 direct outcomes + 12 trust sub-items = 24 outcomes total
# Then compute subscales and trust_multidimensional in build_cell_groups_from_store.py

TIER2_OUTCOMES_WITH_ITEMS = {
    # 12 direct Tier-2 outcomes (all except trust_multidimensional)
    "trust_post",
    "distrust_post",
    "funding_perceptions",
    "policy_role_mean",
    "inst_trust_mean",
    "belief_post",
    "concern_mean",
    "policy_general",
    "policy_specific_mean",
    "behavior_mean",
    "donation_ams",
    "newsletter_signup",
    # 12 trust sub-items (used to compute 4 trust subscales)
    "trust_competence_1",
    "trust_competence_2",
    "trust_competence_3",
    "trust_integrity_1",
    "trust_integrity_2",
    "trust_integrity_3",
    "trust_benevolence_1",
    "trust_benevolence_2",
    "trust_benevolence_3",
    "trust_openness_1",
    "trust_openness_2",
    "trust_openness_3",
}


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
    log.info("GROUP PROMPTS BY TIER-2 MAIN CELLS")
    log.info("(condition × outcome, including 12 trust sub-items for extraction)")
    log.info("=" * 78)

    # Load items metadata
    log.info("Loading items_meta.json...")
    with open(output_dir / "items_meta.json") as f:
        items_meta = json.load(f)
    log.info(f"Loaded {len(items_meta)} items")

    # Group prompts by (condition, outcome) — 12 direct + 12 trust sub-items
    log.info("Grouping prompts by (condition × 24 outcomes with items)...")
    cells = defaultdict(list)
    n_prompts = 0
    skipped_items = 0

    with open(output_dir / "prompts.jsonl") as f:
        for line in f:
            profile = json.loads(line)
            profile_id = profile["profile_id"]
            condition = profile["condition"]

            # For each item, add to the appropriate cell (if Tier-2 outcome or trust sub-item)
            for item_meta in items_meta:
                item_key = item_meta["item_key"]
                submission_var = item_meta["submission_var"]

                # Only include if this is a Tier-2 outcome WITH items (or trust sub-item)
                if submission_var not in TIER2_OUTCOMES_WITH_ITEMS:
                    # Skip items not used in Tier-2 (individual Tier-1 only)
                    skipped_items += 1
                    continue

                # Cell key: condition__outcome
                cell_key = f"{condition}__{submission_var}"

                cells[cell_key].append({
                    "profile_id": profile_id,
                    "item_key": item_key,
                    "submission_var": submission_var,
                })

                n_prompts += 1

    log.info(f"Grouped {n_prompts:,} prompts into {len(cells)} cells")
    log.info(f"(Skipped {skipped_items:,} other items not used in Tier-2)")

    # Write output
    log.info("Writing output...")
    output_path = output_dir / "cells_main_prompts_tier2.json"
    with open(output_path, "w") as f:
        json.dump(cells, f, indent=2)
    log.info(f"Wrote {output_path}")

    # Summary statistics
    log.info("-" * 78)
    log.info("SUMMARY STATISTICS")
    log.info("-" * 78)
    log.info(f"Total cells: {len(cells)}")
    log.info(f"Total prompts grouped: {n_prompts:,}")
    log.info(f"Outcomes with items: {len(TIER2_OUTCOMES_WITH_ITEMS)}")

    # Check structure
    log.info("\nExpected structure:")
    log.info(f"  Conditions: 17")
    log.info(f"  Outcomes with items: 24 (12 direct + 12 trust sub-items)")
    log.info(f"  Expected cells: 17 × 24 = 408")
    log.info(f"  Actual cells: {len(cells)}")

    if len(cells) == 408:
        log.info("  ✓ Cell count matches expected")
    else:
        log.warning(f"  ✗ Expected 408 cells, got {len(cells)}")

    # List all outcomes
    log.info("\nOutcomes with items (24 total):")
    log.info("  Direct Tier-2 outcomes (12):")
    direct_outcomes = sorted([o for o in TIER2_OUTCOMES_WITH_ITEMS if not o.startswith("trust_") or o == "trust_post"])
    for outcome in direct_outcomes:
        log.info(f"    - {outcome}")
    log.info("  Trust sub-items for subscale computation (12):")
    trust_items = sorted([o for o in TIER2_OUTCOMES_WITH_ITEMS if o.startswith("trust_") and o != "trust_post"])
    for outcome in trust_items:
        log.info(f"    - {outcome}")

    # Sample output
    log.info("\nSample cells (first 5):")
    for i, (key, prompts) in enumerate(sorted(cells.items())[:5]):
        log.info(f"  {key}: {len(prompts)} prompts")

    log.info("\n" + "=" * 78)
    log.info("✓ CELLS MAIN PROMPTS (TIER-2) GENERATED")
    log.info("=" * 78)
    log.info("\nNote: This extracts 24 outcomes (12 direct + 12 trust sub-items).")
    log.info("The 12 trust sub-items are used to compute trust_multidimensional,")
    log.info("which is computed in build_cell_groups_from_store.py.")


if __name__ == "__main__":
    setup_logging()
    main()
