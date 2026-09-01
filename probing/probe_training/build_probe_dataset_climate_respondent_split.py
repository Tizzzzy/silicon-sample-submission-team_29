#!/usr/bin/env python3
"""
Build probe dataset from climate representations with RESPONDENT-LEVEL splitting to avoid data leakage.

Workflow:
1. Load extracted_representations_climate.json (20,031 individual entries, flat list)
2. Split respondents (case_ID) into train/test FIRST (80/20 by respondent)
3. Build SEPARATE probe datasets for train and test:
   - Train: groups formed only from train respondents
   - Test: groups formed only from test respondents
   - Zero individual overlap between train and test
4. Output separate NPZ files + metadata for each

This prevents data leakage where the same respondent appears in overlapping
groups in both train and test sets.

Key differences from politic dataset:
- Input is a flat JSON list, not wrapped in {"representations": [...]}
- Demographics are separate top-level fields, not a composite group_id string
- No separate "condition" field → use constant placeholder "N/A"
- Representation field is "last_token_residual_stream", not "representation"
- 9 demographic dimensions (age_bin, gender, race, etc.) instead of 4
- MIN_N = 2 instead of 3 (sparser data: ~1.24 entries/respondent vs ~4.97)
"""

import json
import numpy as np
from collections import defaultdict
from itertools import combinations
import time
import warnings
import sys
import os

# For large JSON files, use ijson for streaming instead of full json.loads
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "../representation_extract/extracted_representations_climate.json"
OUTPUT_TRAIN_NPZ = "probe_dataset_climate_train.npz"
OUTPUT_TEST_NPZ = "probe_dataset_climate_test.npz"
OUTPUT_TRAIN_METADATA = "probe_dataset_climate_train_metadata.json"
OUTPUT_TEST_METADATA = "probe_dataset_climate_test_metadata.json"
DIMENSIONS = ["age_bin", "gender", "race", "educ_category", "income_category",
              "region4", "party", "ideology", "religion"]
MIN_N = 2
REP_DIM = 5120
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
CONDITION_PLACEHOLDER = "N/A"

# task_type to question_id/label mapping (from prompt_creation.py's question_bank)
TASK_TYPE_TO_QID = {
    "happening": 0,
    "cause": 1,
    "worry": 2,
    "priority": 3,
}
QID_TO_LABEL = {
    0: "Do you think that global warming is happening?",
    1: "Assuming global warming is happening, what do you think is the primary cause?",
    2: "How worried are you about global warming?",
    3: "Do you think that global warming should be a priority for the president and Congress?",
}

print("="*80)
print("Build Probe Dataset with Respondent-Level Split (Climate) (NO LEAKAGE)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Input file: {INPUT_FILE}")
print(f"  Train NPZ: {OUTPUT_TRAIN_NPZ}")
print(f"  Test NPZ: {OUTPUT_TEST_NPZ}")
print(f"  Train ratio: {TRAIN_RATIO}")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Min group size: {MIN_N}")
print(f"  Dimensions: {len(DIMENSIONS)}")

# ============================================================================
# Age binning helper
# ============================================================================

def bin_age(age):
    """Bin continuous age (float) into standard demographic brackets."""
    try:
        age = float(age)
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert age to float: {age!r}")

    if age < 35:
        return "18-34"
    elif age < 55:
        return "35-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"

# ============================================================================
# Generate all non-empty dimension subsets
# ============================================================================

def generate_dimension_subsets(dimensions=DIMENSIONS):
    """Generate all non-empty subsets of dimensions."""
    subsets = []
    for r in range(1, len(dimensions) + 1):
        for combo in combinations(dimensions, r):
            subsets.append(combo)
    return subsets

subsets = generate_dimension_subsets()
assert len(subsets) == 511, f"Expected 511 subsets, got {len(subsets)}"
print(f"  Generated {len(subsets)} dimension subsets (2^9 - 1)")

# ============================================================================
# PASS 1: Stream JSON, identify respondents, and do train/test split
# ============================================================================

print(f"\nPass 1: Streaming {INPUT_FILE} to identify respondents...")
t0 = time.time()

all_respondent_ids = set()
entry_count = 0

# First pass: just count unique respondents (memory-efficient)
if HAS_IJSON:
    with open(INPUT_FILE, 'r') as f:
        for i, entry in enumerate(ijson.items(f, 'item')):
            if (i + 1) % 5000 == 0:
                print(f"  Counting entry {i+1}...", flush=True)
            respondent_id = int(round(entry['case_ID']))
            all_respondent_ids.add(respondent_id)
            entry_count += 1
else:
    print(f"  ijson not available, attempting full json.loads (may be memory-intensive)...")
    with open(INPUT_FILE, 'r') as f:
        content = f.read()
        data = json.loads(content)
    for entry in data:
        respondent_id = int(round(entry['case_ID']))
        all_respondent_ids.add(respondent_id)
        entry_count += 1

print(f"Pass 1 complete in {time.time() - t0:.1f}s")
print(f"  Total entries: {entry_count}")
print(f"  Unique respondents: {len(all_respondent_ids)}")

# Split respondents into train/test
np.random.seed(RANDOM_SEED)
all_respondent_ids = sorted(list(all_respondent_ids))
n_train = int(len(all_respondent_ids) * TRAIN_RATIO)
train_respondent_ids = set(np.random.choice(all_respondent_ids, n_train, replace=False))
test_respondent_ids = set(all_respondent_ids) - train_respondent_ids

# Explicit zero-overlap check
assert len(train_respondent_ids & test_respondent_ids) == 0, "Train/test respondents overlap!"

print(f"\nRespondent split (seed={RANDOM_SEED}):")
print(f"  Train respondents: {len(train_respondent_ids)}")
print(f"  Test respondents:  {len(test_respondent_ids)}")

# ============================================================================
# PASS 2: Stream JSON again, build finest buckets separately for train/test
# ============================================================================

print(f"\nPass 2: Building finest buckets by streaming JSON again...")

finest_train = {}
finest_test = {}
n_train_entries = 0
n_test_entries = 0

t0 = time.time()

if HAS_IJSON:
    with open(INPUT_FILE, 'r') as f:
        for i, entry in enumerate(ijson.items(f, 'item')):
            if (i + 1) % 5000 == 0:
                print(f"  Processing entry {i+1}...", flush=True)

            respondent_id = int(round(entry['case_ID']))
            qid = TASK_TYPE_TO_QID[entry["task_type"]]
            age_bin = bin_age(entry["age"])

            key = (
                qid,
                CONDITION_PLACEHOLDER,
                age_bin,
                entry["gender"],
                entry["race"],
                entry["educ_category"],
                entry["income_category"],
                entry["region4"],
                entry["party"],
                entry["ideology"],
                entry["religion"],
            )

            vec = np.asarray(entry["last_token_residual_stream"], dtype=np.float64)

            # Decide which bucket dict to use
            if respondent_id in train_respondent_ids:
                finest = finest_train
                n_train_entries += 1
            else:
                finest = finest_test
                n_test_entries += 1

            b = finest.get(key)
            if b is None:
                b = {
                    "sum_vec": np.zeros(REP_DIM, dtype=np.float64),
                    "sum_opinion": 0.0,
                    "n": 0,
                    "question_label": QID_TO_LABEL[qid],
                }
                finest[key] = b

            b["sum_vec"] += vec
            b["sum_opinion"] += entry["target"]
            b["n"] += 1

else:
    # Fallback: full json.loads (less memory-efficient)
    with open(INPUT_FILE, 'r') as f:
        content = f.read()
        data = json.loads(content)

    for entry in data:
        respondent_id = int(round(entry['case_ID']))
        qid = TASK_TYPE_TO_QID[entry["task_type"]]
        age_bin = bin_age(entry["age"])

        key = (
            qid,
            CONDITION_PLACEHOLDER,
            age_bin,
            entry["gender"],
            entry["race"],
            entry["educ_category"],
            entry["income_category"],
            entry["region4"],
            entry["party"],
            entry["ideology"],
            entry["religion"],
        )

        vec = np.asarray(entry["last_token_residual_stream"], dtype=np.float64)

        if respondent_id in train_respondent_ids:
            finest = finest_train
            n_train_entries += 1
        else:
            finest = finest_test
            n_test_entries += 1

        b = finest.get(key)
        if b is None:
            b = {
                "sum_vec": np.zeros(REP_DIM, dtype=np.float64),
                "sum_opinion": 0.0,
                "n": 0,
                "question_label": QID_TO_LABEL[qid],
            }
            finest[key] = b

        b["sum_vec"] += vec
        b["sum_opinion"] += entry["target"]
        b["n"] += 1

print(f"Pass 2 complete in {time.time() - t0:.1f}s")
print(f"  Train: {len(finest_train)} finest buckets, {n_train_entries} entries")
print(f"  Test:  {len(finest_test)} finest buckets, {n_test_entries} entries")

# Sanity check
print(f"\nSanity check - entry counts:")
print(f"  Train: {n_train_entries}")
print(f"  Test:  {n_test_entries}")

# ============================================================================
# PASS 3: Aggregate by dimension subsets for train and test
# ============================================================================

print(f"\nPass 3: Aggregating by {len(subsets)} dimension subsets...")

def build_rows_from_finest(finest, set_name):
    """Build probe rows from finest buckets."""
    rows = []

    for subset_idx, dim_subset in enumerate(subsets):
        agg = {}
        for (qid, cond, age_bin, gender, race, educ, inc, reg, party, ideo, relig), b in finest.items():
            full_dims = {
                "age_bin": age_bin,
                "gender": gender,
                "race": race,
                "educ_category": educ,
                "income_category": inc,
                "region4": reg,
                "party": party,
                "ideology": ideo,
                "religion": relig,
            }
            proj = tuple(full_dims[d] for d in dim_subset)
            agg_key = (qid, cond, proj)

            a = agg.get(agg_key)
            if a is None:
                a = {
                    "sum_vec": np.zeros(REP_DIM, dtype=np.float64),
                    "sum_opinion": 0.0,
                    "n": 0,
                    "question_label": b["question_label"],
                }
                agg[agg_key] = a

            a["sum_vec"] += b["sum_vec"]
            a["sum_opinion"] += b["sum_opinion"]
            a["n"] += b["n"]

        n_before = len(agg)
        for (qid, cond, proj), a in agg.items():
            if a["n"] < MIN_N:
                continue

            mean_vec = (a["sum_vec"] / a["n"]).astype(np.float32)
            mean_opinion = a["sum_opinion"] / a["n"]

            rows.append({
                "X": mean_vec,
                "y": mean_opinion,
                "n": a["n"],
                "n_dims": len(dim_subset),
                "dimension_subset": list(dim_subset),
                "dimension_values": dict(zip(dim_subset, proj)),
                "condition": cond,
                "question_id": qid,
                "question_label": a["question_label"],
            })

        n_after = sum(1 for (qid, cond, proj), a in agg.items() if a["n"] >= MIN_N)
        if subset_idx % 25 == 0:
            print(f"  [{set_name}] Subset {subset_idx+1:3d}/{len(subsets)}: {len(dim_subset)}-D, "
                  f"{n_before} groups, {n_after} after MIN_N={MIN_N}")

    return rows

rows_train = build_rows_from_finest(finest_train, "Train")
rows_test = build_rows_from_finest(finest_test, "Test")

print(f"\nTotal rows after filtering:")
print(f"  Train: {len(rows_train)}")
print(f"  Test:  {len(rows_test)}")

if len(rows_train) == 0 or len(rows_test) == 0:
    raise ValueError("No rows passed MIN_N filter!")

# ============================================================================
# Prepare output arrays
# ============================================================================

print(f"\nPreparing output arrays...")

def prepare_arrays(rows):
    """Prepare numpy arrays from rows."""
    X = np.stack([r["X"] for r in rows]).astype(np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    n_dims_arr = np.array([r["n_dims"] for r in rows], dtype=np.int8)
    question_id_arr = np.array([r["question_id"] for r in rows], dtype=np.int16)

    return X, y, n_dims_arr, question_id_arr

X_train, y_train, n_dims_train, question_id_train = prepare_arrays(rows_train)
X_test, y_test, n_dims_test, question_id_test = prepare_arrays(rows_test)

# Condition codes (build unified map from all rows)
all_conditions = sorted(set([r["condition"] for r in rows_train] + [r["condition"] for r in rows_test]))
cond_to_code = {c: i for i, c in enumerate(all_conditions)}
condition_code_train = np.array([cond_to_code[r["condition"]] for r in rows_train], dtype=np.int8)
condition_code_test = np.array([cond_to_code[r["condition"]] for r in rows_test], dtype=np.int8)

print(f"  Train X: {X_train.shape}, y: {y_train.shape}")
print(f"  Test X:  {X_test.shape}, y: {y_test.shape}")
print(f"  Conditions: {all_conditions}")

# ============================================================================
# Save outputs
# ============================================================================

print(f"\nSaving train set...")
np.savez_compressed(
    OUTPUT_TRAIN_NPZ,
    X=X_train, y=y_train, n_dims=n_dims_train, question_id=question_id_train, condition_code=condition_code_train
)
train_size_mb = os.path.getsize(OUTPUT_TRAIN_NPZ) / 1024 / 1024
print(f"  Saved {len(rows_train)} rows to {OUTPUT_TRAIN_NPZ} ({train_size_mb:.1f} MB)")

print(f"\nSaving test set...")
np.savez_compressed(
    OUTPUT_TEST_NPZ,
    X=X_test, y=y_test, n_dims=n_dims_test, question_id=question_id_test, condition_code=condition_code_test
)
test_size_mb = os.path.getsize(OUTPUT_TEST_NPZ) / 1024 / 1024
print(f"  Saved {len(rows_test)} rows to {OUTPUT_TEST_NPZ} ({test_size_mb:.1f} MB)")

# ============================================================================
# Save metadata
# ============================================================================

def save_metadata(rows, output_file, set_name, respondent_ids):
    """Save metadata for a dataset."""
    meta = {
        "set_name": set_name,
        "n_rows": len(rows),
        "n_respondents": len(respondent_ids),
        "min_n": MIN_N,
        "dimensions": DIMENSIONS,
        "condition_code_map": cond_to_code,
        "n_dims_distribution": {
            str(nd): int(sum(1 for r in rows if r["n_dims"] == nd))
            for nd in range(1, len(DIMENSIONS) + 1)
        },
        "rows": [{
            "row_index": i,
            "n": r["n"],
            "n_dims": r["n_dims"],
            "dimension_subset": r["dimension_subset"],
            "dimension_values": r["dimension_values"],
            "condition": r["condition"],
            "question_id": r["question_id"],
            "question_label": r["question_label"],
            "y_mean": float(r["y"]),
        } for i, r in enumerate(rows)],
    }

    with open(output_file, 'w') as f:
        json.dump(meta, f, indent=2)

print(f"Saving train metadata...")
save_metadata(rows_train, OUTPUT_TRAIN_METADATA, "train", train_respondent_ids)

print(f"Saving test metadata...")
save_metadata(rows_test, OUTPUT_TEST_METADATA, "test", test_respondent_ids)

# ============================================================================
# Summary statistics
# ============================================================================

print(f"\n" + "="*80)
print("Summary Statistics")
print("="*80)

for set_name, y_set, rows_set in [("Train", y_train, rows_train), ("Test", y_test, rows_test)]:
    print(f"\n{set_name} set:")
    print(f"  Rows: {len(rows_set)}")

    print(f"  Granularity distribution:")
    for nd in range(1, len(DIMENSIONS) + 1):
        c = sum(1 for r in rows_set if r["n_dims"] == nd)
        pct = 100*c/len(rows_set) if len(rows_set) > 0 else 0
        print(f"    {nd}D: {c} ({pct:.1f}%)")

    print(f"  Condition distribution:")
    for cond in all_conditions:
        c = sum(1 for r in rows_set if r["condition"] == cond)
        pct = 100*c/len(rows_set) if len(rows_set) > 0 else 0
        print(f"    {cond:25s}: {c} ({pct:.1f}%)")

    print(f"  Ground truth (y): [{y_set.min():.3f}, {y_set.max():.3f}], mean={y_set.mean():.3f}, std={y_set.std():.3f}")

    ns = np.array([r["n"] for r in rows_set])
    print(f"  Group sizes: [{ns.min()}, {ns.max()}], mean={ns.mean():.1f}, std={ns.std():.1f}")

print(f"\n✅ Dataset build complete with NO LEAKAGE!")
print(f"   Train & test sets use disjoint sets of respondents")
print(f"   Ready for training: {OUTPUT_TRAIN_NPZ} + {OUTPUT_TEST_NPZ}")
