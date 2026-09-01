#!/usr/bin/env python3
"""
Build probe dataset with RESPONDENT-LEVEL splitting to avoid data leakage.

Workflow:
1. Load extracted_representations_politic.json (22,409 individual entries)
2. Split respondents into train/test FIRST (80/20 by respondent_id)
3. Build SEPARATE probe datasets for train and test:
   - Train: groups formed only from train respondents
   - Test: groups formed only from test respondents
   - Zero individual overlap between train and test
4. Output separate NPZ files + metadata for each

This prevents data leakage where the same respondent appears in overlapping
groups in both train and test sets.
"""

import json
import numpy as np
import re
from collections import defaultdict
from itertools import combinations
import time
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "../representation_extract/extracted_representations_politic.json"
OUTPUT_TRAIN_NPZ = "probe_dataset_train.npz"
OUTPUT_TEST_NPZ = "probe_dataset_test.npz"
OUTPUT_TRAIN_METADATA = "probe_dataset_train_metadata.json"
OUTPUT_TEST_METADATA = "probe_dataset_test_metadata.json"
DIMENSIONS = ["education", "gender", "party", "political_knowledge"]
MIN_N = 3
REP_DIM = 5120
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

print("="*80)
print("Build Probe Dataset with Respondent-Level Split (NO LEAKAGE)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Input file: {INPUT_FILE}")
print(f"  Train NPZ: {OUTPUT_TRAIN_NPZ}")
print(f"  Test NPZ: {OUTPUT_TEST_NPZ}")
print(f"  Train ratio: {TRAIN_RATIO}")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Min group size: {MIN_N}")

# ============================================================================
# Parse group_id using regex
# ============================================================================

GROUP_ID_RE = re.compile(
    r"^education_(?P<education>.+)_gender_(?P<gender>.+)"
    r"_party_(?P<party>.+)_political_knowledge_(?P<political_knowledge>.+)$"
)

def parse_group_id(group_id):
    """Parse group_id string into demographic dimensions dict."""
    m = GROUP_ID_RE.match(group_id)
    if not m:
        raise ValueError(f"group_id does not match expected pattern: {group_id!r}")
    dims = m.groupdict()
    rebuilt = "_".join(f"{d}_{dims[d]}" for d in DIMENSIONS)
    assert rebuilt == group_id, f"round-trip mismatch: {rebuilt!r} != {group_id!r}"
    return dims

# ============================================================================
# Generate all non-empty dimension subsets
# ============================================================================

def generate_dimension_subsets(dimensions=DIMENSIONS):
    """Generate all 15 non-empty subsets of dimensions."""
    subsets = []
    for r in range(1, len(dimensions) + 1):
        for combo in combinations(dimensions, r):
            subsets.append(combo)
    return subsets

subsets = generate_dimension_subsets()
assert len(subsets) == 15, f"Expected 15 subsets, got {len(subsets)}"

# ============================================================================
# PASS 1: Load JSON streaming and partition by respondent
# ============================================================================

print(f"\nPass 1: Loading {INPUT_FILE} and partitioning by respondent...")
t0 = time.time()

# Store all entries indexed by respondent_id
entries_by_respondent = defaultdict(list)
all_respondent_ids = set()
entry_count = 0

with open(INPUT_FILE, 'r') as f:
    content = f.read()
    data = json.loads(content)
    representations_data = data['representations']

    for i, entry in enumerate(representations_data):
        if (i + 1) % 5000 == 0:
            print(f"  Processing entry {i+1}/{len(representations_data)}", flush=True)

        respondent_id = entry['respondent_id']
        entries_by_respondent[respondent_id].append(entry)
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

print(f"\nRespondent split (seed={RANDOM_SEED}):")
print(f"  Train respondents: {len(train_respondent_ids)}")
print(f"  Test respondents:  {len(test_respondent_ids)}")

n_train_entries = sum(len(entries_by_respondent[rid]) for rid in train_respondent_ids)
n_test_entries = sum(len(entries_by_respondent[rid]) for rid in test_respondent_ids)
print(f"  Train entries: {n_train_entries} ({100*n_train_entries/entry_count:.1f}%)")
print(f"  Test entries:  {n_test_entries} ({100*n_test_entries/entry_count:.1f}%)")

# ============================================================================
# PASS 2: Build finest buckets separately for train and test
# ============================================================================

print(f"\nPass 2: Building finest buckets for train and test sets...")

def build_finest_buckets(respondent_ids, entries_by_respondent):
    """Build finest-grain buckets from a set of respondents."""
    finest = {}

    for respondent_id in respondent_ids:
        for entry in entries_by_respondent[respondent_id]:
            dims = parse_group_id(entry["group_id"])
            key = (
                entry["question_id"],
                entry["condition"],
                dims["education"],
                dims["gender"],
                dims["party"],
                dims["political_knowledge"],
            )

            vec = np.asarray(entry["representation"], dtype=np.float64)

            b = finest.get(key)
            if b is None:
                b = {
                    "sum_vec": np.zeros(REP_DIM, dtype=np.float64),
                    "sum_opinion": 0.0,
                    "n": 0,
                    "question_label": entry["question_label"],
                }
                finest[key] = b

            b["sum_vec"] += vec
            b["sum_opinion"] += entry["individual_opinion"]
            b["n"] += 1

    return finest

finest_train = build_finest_buckets(train_respondent_ids, entries_by_respondent)
finest_test = build_finest_buckets(test_respondent_ids, entries_by_respondent)

print(f"  Train: {len(finest_train)} finest buckets")
print(f"  Test:  {len(finest_test)} finest buckets")

# Sanity check: entry counts
total_n_train = sum(b["n"] for b in finest_train.values())
total_n_test = sum(b["n"] for b in finest_test.values())
print(f"\nSanity check - entry counts:")
print(f"  Train: {total_n_train} (expected {n_train_entries})")
print(f"  Test:  {total_n_test} (expected {n_test_entries})")
assert total_n_train == n_train_entries and total_n_test == n_test_entries, "Mismatch!"

# ============================================================================
# PASS 3: Aggregate by dimension subsets for train and test
# ============================================================================

print(f"\nPass 3: Aggregating by {len(subsets)} dimension subsets...")

def build_rows_from_finest(finest, set_name):
    """Build probe rows from finest buckets."""
    rows = []

    for subset_idx, dim_subset in enumerate(subsets):
        agg = {}
        for (qid, cond, ed, gen, par, pk), b in finest.items():
            full_dims = {
                "education": ed,
                "gender": gen,
                "party": par,
                "political_knowledge": pk,
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
        if subset_idx % 3 == 0:
            print(f"  [{set_name}] Subset {subset_idx+1:2d}/{len(subsets)}: {len(dim_subset)}-D, "
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
            for nd in range(1, 5)
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
    for nd in range(1, 5):
        c = sum(1 for r in rows_set if r["n_dims"] == nd)
        print(f"    {nd}D: {c} ({100*c/len(rows_set):.1f}%)")

    print(f"  Condition distribution:")
    for cond in all_conditions:
        c = sum(1 for r in rows_set if r["condition"] == cond)
        print(f"    {cond:25s}: {c} ({100*c/len(rows_set):.1f}%)")

    print(f"  Ground truth (y): [{y_set.min():.3f}, {y_set.max():.3f}], mean={y_set.mean():.3f}, std={y_set.std():.3f}")

    ns = np.array([r["n"] for r in rows_set])
    print(f"  Group sizes: [{ns.min()}, {ns.max()}], mean={ns.mean():.1f}, std={ns.std():.1f}")

print(f"\n✅ Dataset build complete with NO LEAKAGE!")
print(f"   Train & test sets use disjoint sets of respondents")
print(f"   Ready for training: {OUTPUT_TRAIN_NPZ} + {OUTPUT_TEST_NPZ}")
