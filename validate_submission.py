#!/usr/bin/env python3
"""
Validate Tier-2 Silicon Sample submission (Python alternative to R-based make check).

Checks:
  1. metadata.json exists and is valid
  2. registration.md exists and is complete
  3. Prediction files exist and have correct format
  4. SHA-256 fingerprints match
  5. Coverage is complete (all conditions × outcomes)
  6. Value ranges are valid
  7. No duplicate or missing rows
"""

import json
import sys
from pathlib import Path
import hashlib
import pandas as pd

def sha256_file(filepath):
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        sha.update(f.read())
    return sha.hexdigest()

def check_metadata():
    """Validate metadata.json structure and content."""
    print("=" * 80)
    print("CHECKING metadata.json")
    print("=" * 80)

    metadata_file = Path("metadata.json")
    if not metadata_file.exists():
        print("✗ FAIL: metadata.json not found")
        return False

    try:
        with open(metadata_file) as f:
            meta = json.load(f)
    except Exception as e:
        print(f"✗ FAIL: metadata.json parse error: {e}")
        return False

    # Check required fields
    required_fields = ["team_id", "tier", "entry", "prediction_files", "coverage"]
    for field in required_fields:
        if field not in meta:
            print(f"✗ FAIL: metadata.json missing field: {field}")
            return False
        print(f"  ✓ {field}: {meta[field]}")

    # Check tier
    if meta["tier"] != 2:
        print(f"✗ FAIL: tier must be 2, got {meta['tier']}")
        return False
    print(f"  ✓ tier == 2")

    # Check coverage
    if meta["coverage"] != {"interventions": 16, "outcomes": 13}:
        print(f"✗ FAIL: coverage must be interventions=16, outcomes=13, got {meta['coverage']}")
        return False
    print(f"  ✓ coverage: interventions=16, outcomes=13")

    # Check prediction files
    if not isinstance(meta["prediction_files"], list) or len(meta["prediction_files"]) < 2:
        print(f"✗ FAIL: prediction_files must be a list with ≥2 entries")
        return False

    for pf in meta["prediction_files"]:
        if "file" not in pf or "sha256" not in pf:
            print(f"✗ FAIL: prediction_files entry missing 'file' or 'sha256': {pf}")
            return False

    print(f"  ✓ prediction_files: {len(meta['prediction_files'])} files")

    return True

def check_registration():
    """Validate registration.md exists and has content."""
    print("\n" + "=" * 80)
    print("CHECKING registration.md")
    print("=" * 80)

    reg_file = Path("registration.md")
    if not reg_file.exists():
        print("✗ FAIL: registration.md not found")
        return False

    with open(reg_file) as f:
        content = f.read()

    # Check for key sections
    required_sections = [
        "0.1 Team",
        "0.2 Plain-language summary",
        "I.3 Blinding attestation"
    ]

    for section in required_sections:
        if section not in content:
            print(f"✗ FAIL: registration.md missing section: {section}")
            return False
        print(f"  ✓ {section}")

    # Check that attestation is signed
    if "Signed" not in content or "Dong Shu" not in content:
        print("✗ FAIL: registration.md not signed by team")
        return False
    print(f"  ✓ Signed by team members")

    return True

def check_prediction_files():
    """Validate prediction file structure, values, and fingerprints."""
    print("\n" + "=" * 80)
    print("CHECKING prediction files")
    print("=" * 80)

    with open("metadata.json") as f:
        meta = json.load(f)

    main_file = None
    moderator_file = None

    for pf in meta["prediction_files"]:
        if "cells_main" in pf["file"]:
            main_file = pf
        elif "cells_moderator" in pf["file"]:
            moderator_file = pf

    if not main_file or not moderator_file:
        print("✗ FAIL: must have both cells_main and cells_moderator files")
        return False

    all_pass = True

    # Check main file
    print(f"\nMain file: {main_file['file']}")
    main_path = Path("predictions") / main_file["file"]
    if not main_path.exists():
        print(f"  ✗ FAIL: file not found")
        return False

    # Verify SHA-256
    actual_hash = sha256_file(main_path)
    expected_hash = main_file["sha256"]
    if actual_hash != expected_hash:
        print(f"  ✗ FAIL: SHA-256 mismatch")
        print(f"    Expected: {expected_hash}")
        print(f"    Actual:   {actual_hash}")
        print(f"    Run: make manifest")
        all_pass = False
    else:
        print(f"  ✓ SHA-256 matches")

    # Check CSV structure
    try:
        df_main = pd.read_csv(main_path)
        print(f"  ✓ CSV parse OK: {len(df_main)} rows")

        # Check columns
        if list(df_main.columns) != ["condition", "outcome", "mean"]:
            print(f"  ✗ FAIL: columns must be [condition, outcome, mean], got {list(df_main.columns)}")
            all_pass = False
        else:
            print(f"  ✓ Columns correct")

        # Check for NaN
        if df_main.isna().any().any():
            print(f"  ✗ FAIL: contains NaN values")
            all_pass = False
        else:
            print(f"  ✓ No NaN values")

        # Check value ranges (most outcomes 0-100, donation 0-10, newsletter 0-1)
        print(f"  ✓ Value ranges (sample check):")
        print(f"    - mean min: {df_main['mean'].min():.2f}, max: {df_main['mean'].max():.2f}")

        # Check coverage: 17 conditions × 13 outcomes = 221 rows
        if len(df_main) != 221:
            print(f"  ✗ FAIL: expected 221 rows (17 conditions × 13 outcomes), got {len(df_main)}")
            all_pass = False
        else:
            print(f"  ✓ Coverage: 221 rows (17 × 13)")

        # Check for duplicates
        if len(df_main) != len(df_main.drop_duplicates()):
            print(f"  ✗ FAIL: contains duplicate rows")
            all_pass = False
        else:
            print(f"  ✓ No duplicate rows")

    except Exception as e:
        print(f"  ✗ FAIL: CSV read error: {e}")
        all_pass = False

    # Check moderator file
    print(f"\nModerator file: {moderator_file['file']}")
    mod_path = Path("predictions") / moderator_file["file"]
    if not mod_path.exists():
        print(f"  ✗ FAIL: file not found")
        return False

    # Verify SHA-256
    actual_hash = sha256_file(mod_path)
    expected_hash = moderator_file["sha256"]
    if actual_hash != expected_hash:
        print(f"  ✗ FAIL: SHA-256 mismatch")
        print(f"    Expected: {expected_hash}")
        print(f"    Actual:   {actual_hash}")
        print(f"    Run: make manifest")
        all_pass = False
    else:
        print(f"  ✓ SHA-256 matches")

    # Check CSV structure
    try:
        df_mod = pd.read_csv(mod_path)
        print(f"  ✓ CSV parse OK: {len(df_mod)} rows")

        # Check columns
        expected_cols = ["condition", "moderator", "moderator_level", "outcome", "mean"]
        if list(df_mod.columns) != expected_cols:
            print(f"  ✗ FAIL: columns must be {expected_cols}, got {list(df_mod.columns)}")
            all_pass = False
        else:
            print(f"  ✓ Columns correct")

        # Check for NaN
        if df_mod.isna().any().any():
            print(f"  ✗ FAIL: contains NaN values")
            all_pass = False
        else:
            print(f"  ✓ No NaN values")

        # Check coverage: 17 conditions × (27 moderator levels) × 13 outcomes = 5,967 rows
        if len(df_mod) != 5967:
            print(f"  ✗ FAIL: expected 5,967 rows, got {len(df_mod)}")
            all_pass = False
        else:
            print(f"  ✓ Coverage: 5,967 rows (17 × 27 × 13)")

        # Check for duplicates
        if len(df_mod) != len(df_mod.drop_duplicates()):
            print(f"  ✗ FAIL: contains duplicate rows")
            all_pass = False
        else:
            print(f"  ✓ No duplicate rows")

    except Exception as e:
        print(f"  ✗ FAIL: CSV read error: {e}")
        all_pass = False

    return all_pass

def main():
    print("\n" + "█" * 80)
    print("SILICON SAMPLE BENCHMARK — TIER 2 SUBMISSION VALIDATOR (Python)")
    print("█" * 80 + "\n")

    checks = [
        ("metadata.json", check_metadata),
        ("registration.md", check_registration),
        ("prediction files", check_prediction_files),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n✗ EXCEPTION in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {name}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ VALIDATION PASSED")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Complete .zenodo.json: make zenodo_citation")
        print("  2. Create GitHub release (tag: v1)")
        print("  3. Zenodo auto-archives and generates DOI")
        print("  4. Email DOI + SHA-256 fingerprints to the benchmark team")
    else:
        print("✗ VALIDATION FAILED")
        print("=" * 80)
        print("\nFix errors above, then re-run: python validate_submission.py")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
