#!/usr/bin/env python3
"""
Generate a synthetic respondent profile pool for the Silicon Sample Benchmark.

This script creates a CSV file containing demographic profiles for synthetic
respondents, distributed across 17 conditions (control + 16 interventions).
Each profile includes: profile_id, condition, and 6 demographic moderators
(gender, age_band, race, education, income, party).

The pool is an INTERMEDIATE ARTIFACT — it contains only the demographic and
condition assignment columns. The 13 outcome variables and 12 trust sub-item
columns must be simulated separately (via LLM calls per profile + intervention).

Total profiles:
  - control: 1,000
  - each intervention (×16): 500
  - total: 9,000 (Tier-1 minimum precision floor per benchmark preregistration)

Demographic distributions:
  - age_band, race: exact Census quotas from preregistration (load-bearing)
  - gender, education, income, party: documented national priors (editable)
  - all moderators drawn independently within each condition (no joint correlation)

Source of truth for exact condition names, moderator levels, and submission
schema: scripts/lib/submission_spec.R (symlink verified; run this generator
from the repo root).

Example usage:
  python scripts/generate_profiles.py --seed 2026 --out synthetic_profiles/profiles_pool.csv
  python scripts/generate_profiles.py --seed 42 --n-control 2000 --n-intervention 1000  # 18,000 total
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONSTANTS — demographic moderators, levels, and target proportions
# ============================================================================

# 17 conditions: control + 16 interventions (order matches submission_spec.R)
CONDITIONS = [
    "control",
    "Corporate reliance",
    "Social justice",
    "Interview Prof. Maraun",
    "Funding",
    "Oil industry misinformation",
    "Measurement & modeling (1)",
    "Former skeptics",
    "High public trust",
    "Measurement & modeling (2)",
    "Peer-review",
    "Scientist community helpers",
    "Consensus",
    "Portrait Prof. Cherry",
    "Model accuracy",
    "Interview Prof. Sebille",
    "Extreme weather predictions",
]

# Moderator level strings (exact match to submission_spec.R or scoring will fail)
MODERATOR_LEVELS = {
    "gender": ["Male", "Female", "Other"],
    "age_band": ["18-29", "30-44", "45-59", "60+"],
    "race": [
        "White / Caucasian",
        "Black / African American",
        "Hispanic / Latino",
        "Asian / Asian American",
        "Other",
    ],
    "education": [
        "Less than high school",
        "High school diploma / GED",
        "Some college or Associate's degree",
        "Bachelor's degree",
        "Master's degree / Professional degree",
        "Doctorate degree / Ph.D.",
    ],
    "income": [
        "Less than $30,000",
        "$30,000 to $55,999",
        "$56,000 to $99,999",
        "$100,000 to $167,999",
        "$168,000 or more",
    ],
    "party": ["Republican", "Democrat", "Independent", "Other"],
}

# Target proportions for each moderator
# Sources noted inline for registration.md D.1 disclosure
MODERATOR_PROBS = {
    # Benchmark preregistration: census-based quotas (load-bearing)
    "age_band": [0.202, 0.260, 0.229, 0.309],  # 18-29, 30-44, 45-59, 60+
    "race": [0.602, 0.123, 0.181, 0.067, 0.027],  # White, Black, Hispanic, Asian, Other

    # National priors (Census ACS, Pew/Gallup) — editable defaults
    "gender": [0.49, 0.49, 0.02],  # Male, Female, Other (survey-panel estimate)
    "education": [
        0.09, 0.27, 0.29, 0.23, 0.09, 0.03
    ],  # Census ACS adults 25+: <HS, HS/GED, Some college, Bachelor's, Master's/Prof., Doctorate
    "income": [
        0.19, 0.20, 0.26, 0.23, 0.12
    ],  # Census ACS households (rebucketed): <$30k, $30–56k, $56–100k, $100–168k, $168k+
    "party": [0.27, 0.27, 0.41, 0.05],  # Gallup aggregate: Rep, Dem, Ind, Other
}


def generate_pool(seed, n_control=1000, n_intervention=500):
    """
    Generate demographic profile pool.

    Args:
        seed (int): Random seed for reproducibility
        n_control (int): Number of profiles in control condition (default 1,000)
        n_intervention (int): Number of profiles per intervention (default 500)

    Returns:
        pd.DataFrame: Synthetic profile pool with columns:
            profile_id, condition, gender, age_band, race, education, income, party
    """
    rng = np.random.default_rng(seed)
    rows = []
    profile_counter = 0

    # Sample each condition independently
    for condition in CONDITIONS:
        n = n_control if condition == "control" else n_intervention

        # Draw each moderator independently for this condition
        samples = {
            "condition": [condition] * n,
            "gender": rng.choice(
                MODERATOR_LEVELS["gender"], size=n, p=MODERATOR_PROBS["gender"]
            ),
            "age_band": rng.choice(
                MODERATOR_LEVELS["age_band"], size=n, p=MODERATOR_PROBS["age_band"]
            ),
            "race": rng.choice(
                MODERATOR_LEVELS["race"], size=n, p=MODERATOR_PROBS["race"]
            ),
            "education": rng.choice(
                MODERATOR_LEVELS["education"],
                size=n,
                p=MODERATOR_PROBS["education"],
            ),
            "income": rng.choice(
                MODERATOR_LEVELS["income"], size=n, p=MODERATOR_PROBS["income"]
            ),
            "party": rng.choice(
                MODERATOR_LEVELS["party"], size=n, p=MODERATOR_PROBS["party"]
            ),
        }

        # Assign profile_id sequentially
        samples["profile_id"] = [
            f"p{profile_counter + i:05d}" for i in range(n)
        ]
        profile_counter += n

        # Append to rows
        for i in range(n):
            rows.append({
                "profile_id": samples["profile_id"][i],
                "condition": samples["condition"][i],
                "gender": samples["gender"][i],
                "age_band": samples["age_band"][i],
                "race": samples["race"][i],
                "education": samples["education"][i],
                "income": samples["income"][i],
                "party": samples["party"][i],
            })

    df = pd.DataFrame(rows)

    # Reorder columns to match submission_spec.R tier1_required[0:8]
    df = df[["profile_id", "condition", "gender", "age_band", "race", "education", "income", "party"]]

    return df


def validate_pool(df):
    """
    Validate profile pool structure and contents.

    Args:
        df (pd.DataFrame): Profile pool to validate

    Raises:
        ValueError: If validation fails
    """
    errors = []

    # Total row count
    if len(df) != 9000:
        errors.append(f"Expected 9,000 rows, got {len(df)}")

    # No missing values
    if df.isnull().any().any():
        errors.append(f"Found {df.isnull().sum().sum()} missing values")

    # Condition set and counts
    expected_counts = {c: (1000 if c == "control" else 500) for c in CONDITIONS}
    actual_counts = df["condition"].value_counts().to_dict()

    for condition in CONDITIONS:
        expected = expected_counts[condition]
        actual = actual_counts.get(condition, 0)
        if actual != expected:
            errors.append(
                f"Condition '{condition}': expected {expected}, got {actual}"
            )

    # Moderator levels
    for moderator, allowed_levels in MODERATOR_LEVELS.items():
        actual_levels = set(df[moderator].unique())
        allowed_set = set(allowed_levels)
        if not actual_levels.issubset(allowed_set):
            unexpected = actual_levels - allowed_set
            errors.append(
                f"Moderator '{moderator}' has unexpected levels: {unexpected}"
            )

    # profile_id uniqueness
    if df["profile_id"].duplicated().any():
        errors.append(f"Found {df['profile_id'].duplicated().sum()} duplicate profile_id values")

    if errors:
        raise ValueError("Validation failed:\n  " + "\n  ".join(errors))


def print_summary(df):
    """
    Print a summary of demographic distribution vs. targets.

    Args:
        df (pd.DataFrame): Profile pool
    """
    print("\n" + "=" * 80)
    print("DEMOGRAPHIC DISTRIBUTION SUMMARY")
    print("=" * 80)

    # Overall distribution
    print("\nOVERALL (across all 9,000 profiles):")
    print("-" * 80)
    for moderator in ["gender", "age_band", "race", "education", "income", "party"]:
        print(f"\n{moderator.upper()}:")
        counts = df[moderator].value_counts()
        for level in MODERATOR_LEVELS[moderator]:
            count = counts.get(level, 0)
            pct = 100 * count / len(df)
            target_pct = 100 * MODERATOR_PROBS[moderator][
                MODERATOR_LEVELS[moderator].index(level)
            ]
            match = "✓" if abs(pct - target_pct) < 0.5 else "~"
            print(f"  {match} {level:45s} {count:5d} ({pct:5.2f}% vs. {target_pct:5.2f}% target)")

    # Per-condition (spot-check: control + first intervention)
    print("\n" + "-" * 80)
    print("PER-CONDITION SPOT-CHECK (Control + 1 intervention shown):\n")
    for condition in ["control", "Corporate reliance"]:
        cond_df = df[df["condition"] == condition]
        n = len(cond_df)
        print(f"{condition} (N={n}):")
        for moderator in ["age_band", "race", "party"]:  # Show top 3
            counts = cond_df[moderator].value_counts()
            for level in MODERATOR_LEVELS[moderator][:2]:  # First 2 levels
                count = counts.get(level, 0)
                pct = 100 * count / n
                target_pct = 100 * MODERATOR_PROBS[moderator][
                    MODERATOR_LEVELS[moderator].index(level)
                ]
                print(f"  {moderator} '{level[:20]:20s}' {count:3d} ({pct:5.2f}%)")

    print("\n" + "=" * 80)
    print(f"Total profiles generated: {len(df):,}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic respondent profile pool for Silicon Sample Benchmark"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility (default: 2026)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="synthetic_profiles/profiles_pool.csv",
        help="Output CSV file path (default: synthetic_profiles/profiles_pool.csv)",
    )
    parser.add_argument(
        "--n-control",
        type=int,
        default=1000,
        help="Number of control profiles (default: 1,000)",
    )
    parser.add_argument(
        "--n-intervention",
        type=int,
        default=500,
        help="Number of profiles per intervention (default: 500; 16 interventions × 500 + 1,000 control = 9,000 total)",
    )

    args = parser.parse_args()

    # Generate
    print(f"Generating profile pool (seed={args.seed})...")
    df = generate_pool(args.seed, n_control=args.n_control, n_intervention=args.n_intervention)

    # Validate
    try:
        validate_pool(df)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print_summary(df)

    # Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Profile pool written to: {out_path}\n")


if __name__ == "__main__":
    main()
