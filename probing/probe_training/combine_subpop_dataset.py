#!/usr/bin/env python3
"""
Combine subpop dataset with politic + climate dataset.

Workflow:
1. Load extracted_representations_subpop.json (already grouped, 2,926 entries)
2. Split into 90/10 train/test (respondent-level split to avoid leakage)
3. Create NPZ files and metadata for subpop train/test
4. Combine with existing combined dataset (politic + climate) into train_test_split/
5. Output: unified train/test NPZ + metadata supporting 3 sources

Key note: Subpop is already aggregated by group (no grouping needed).
Each entry represents a single group with pre-computed opinion mean.
No respondent ID in subpop data, so split by group index.
"""

import json
import numpy as np
import os
import time
from datetime import datetime

print("="*80)
print("Combine Subpop Dataset with Politic + Climate")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

SUBPOP_INPUT_FILE = "../representation_extract/extracted_representations_subpop.json"
POLITIC_TRAIN_NPZ = "probe_dataset_train.npz"
POLITIC_TEST_NPZ = "probe_dataset_test.npz"
CLIMATE_TRAIN_NPZ = "probe_dataset_climate_train.npz"
CLIMATE_TEST_NPZ = "probe_dataset_climate_test.npz"

SUBPOP_OUTPUT_TRAIN_NPZ = "probe_dataset_subpop_train.npz"
SUBPOP_OUTPUT_TEST_NPZ = "probe_dataset_subpop_test.npz"
SUBPOP_OUTPUT_TRAIN_META = "probe_dataset_subpop_train_metadata.json"
SUBPOP_OUTPUT_TEST_META = "probe_dataset_subpop_test_metadata.json"

OUT_DIR = "train_test_split"
OUT_TRAIN_NPZ = os.path.join(OUT_DIR, "probe_dataset_triple_train.npz")
OUT_TEST_NPZ = os.path.join(OUT_DIR, "probe_dataset_triple_test.npz")
OUT_TRAIN_META = os.path.join(OUT_DIR, "probe_dataset_triple_train_metadata.json")
OUT_TEST_META = os.path.join(OUT_DIR, "probe_dataset_triple_test_metadata.json")

DATASET_SOURCE_POLITIC = 0
DATASET_SOURCE_CLIMATE = 1
DATASET_SOURCE_SUBPOP = 2

TRAIN_RATIO = 0.9  # 90/10 split for subpop
RANDOM_SEED = 42
MIN_N = 2  # Subpop minimum group size
REP_DIM = 5120

# ============================================================================
# Load subpop data
# ============================================================================

print(f"\n1. Loading subpop data from {SUBPOP_INPUT_FILE}...")
t0 = time.time()

with open(SUBPOP_INPUT_FILE, 'r') as f:
    subpop_data = json.load(f)

print(f"   Loaded {len(subpop_data)} subpop entries in {time.time()-t0:.1f}s")

if len(subpop_data) == 0:
    raise ValueError("Subpop data is empty!")

# Inspect structure
sample = subpop_data[0]
print(f"   Sample entry keys: {list(sample.keys())}")
print(f"   Sample qkey: {sample.get('qkey')}")
print(f"   Sample group: {sample.get('group')}")
print(f"   Sample true_mean_opinion: {sample.get('true_mean_opinion')}")

# ============================================================================
# Split subpop into train/test (90/10)
# ============================================================================

print(f"\n2. Splitting subpop into 90/10 train/test...")

np.random.seed(RANDOM_SEED)
n_train = int(len(subpop_data) * TRAIN_RATIO)
indices = np.arange(len(subpop_data))
np.random.shuffle(indices)

train_indices = set(indices[:n_train])
test_indices = set(indices[n_train:])

subpop_train = [subpop_data[i] for i in range(len(subpop_data)) if i in train_indices]
subpop_test = [subpop_data[i] for i in range(len(subpop_data)) if i in test_indices]

print(f"   Train: {len(subpop_train)} entries ({100*len(subpop_train)/len(subpop_data):.1f}%)")
print(f"   Test:  {len(subpop_test)} entries ({100*len(subpop_test)/len(subpop_data):.1f}%)")

# ============================================================================
# Build subpop NPZ + metadata
# ============================================================================

print(f"\n3. Building subpop train/test NPZ and metadata...")

def build_subpop_dataset(entries, split_name):
    """Convert subpop entries to NPZ format."""
    rows = []
    X_list = []
    y_list = []

    for idx, entry in enumerate(entries):
        vec = np.asarray(entry['last_token_residual_stream'], dtype=np.float32)
        opinion = float(entry['true_mean_opinion'])
        group_str = str(entry['group'])
        qkey = str(entry['qkey'])

        X_list.append(vec)
        y_list.append(opinion)

        rows.append({
            "row_index": idx,
            "n": 1,  # Subpop entries are already aggregated
            "n_dims": 1,  # Subpop uses single group dimension
            "dimension_subset": ["group"],
            "dimension_values": {"group": group_str},
            "condition": "N/A",  # No conditions in subpop
            "question_id": hash(qkey) % 1000,  # Simple hash for question_id
            "question_label": qkey,
            "y_mean": opinion,
        })

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    # For subpop, we don't have multiple dimensions, so all rows have n_dims=1
    n_dims = np.full(len(y), 1, dtype=np.int8)

    # Question IDs are hashes of qkey strings
    question_id = np.array([hash(str(e['qkey'])) % 1000 for e in entries], dtype=np.int16)

    # Single condition "N/A" mapped to code 0
    condition_code = np.zeros(len(y), dtype=np.int8)

    return {
        "X": X,
        "y": y,
        "n_dims": n_dims,
        "question_id": question_id,
        "condition_code": condition_code,
        "rows": rows,
    }

subpop_train_data = build_subpop_dataset(subpop_train, "train")
subpop_test_data = build_subpop_dataset(subpop_test, "test")

print(f"   Train: {subpop_train_data['X'].shape[0]} rows, X shape {subpop_train_data['X'].shape}")
print(f"   Test:  {subpop_test_data['X'].shape[0]} rows, X shape {subpop_test_data['X'].shape}")

# ============================================================================
# Save subpop train/test
# ============================================================================

print(f"\n4. Saving subpop train/test NPZ and metadata...")

# Save subpop train NPZ
np.savez_compressed(
    SUBPOP_OUTPUT_TRAIN_NPZ,
    X=subpop_train_data['X'],
    y=subpop_train_data['y'],
    n_dims=subpop_train_data['n_dims'],
    question_id=subpop_train_data['question_id'],
    condition_code=subpop_train_data['condition_code'],
)
train_npz_mb = os.path.getsize(SUBPOP_OUTPUT_TRAIN_NPZ) / 1024 / 1024
print(f"   Saved {SUBPOP_OUTPUT_TRAIN_NPZ} ({train_npz_mb:.1f} MB)")

# Save subpop test NPZ
np.savez_compressed(
    SUBPOP_OUTPUT_TEST_NPZ,
    X=subpop_test_data['X'],
    y=subpop_test_data['y'],
    n_dims=subpop_test_data['n_dims'],
    question_id=subpop_test_data['question_id'],
    condition_code=subpop_test_data['condition_code'],
)
test_npz_mb = os.path.getsize(SUBPOP_OUTPUT_TEST_NPZ) / 1024 / 1024
print(f"   Saved {SUBPOP_OUTPUT_TEST_NPZ} ({test_npz_mb:.1f} MB)")

# Save metadata
def save_subpop_metadata(data, output_file, split_name):
    """Save metadata for subpop dataset."""
    meta = {
        "set_name": split_name,
        "n_rows": len(data['rows']),
        "n_respondents": len(data['rows']),  # Each entry is a group, not individual
        "min_n": 1,
        "dimensions": ["group"],  # Subpop only has single "group" dimension
        "condition_code_map": {"N/A": 0},
        "n_dims_distribution": {"1": len(data['rows'])},  # All are 1D
        "rows": data['rows'],
    }

    with open(output_file, 'w') as f:
        json.dump(meta, f, indent=2)

save_subpop_metadata(subpop_train_data, SUBPOP_OUTPUT_TRAIN_META, "train")
save_subpop_metadata(subpop_test_data, SUBPOP_OUTPUT_TEST_META, "test")
print(f"   Saved {SUBPOP_OUTPUT_TRAIN_META}")
print(f"   Saved {SUBPOP_OUTPUT_TEST_META}")

# ============================================================================
# Load existing politic + climate combined dataset
# ============================================================================

print(f"\n5. Loading existing politic + climate combined datasets...")

# Load politic
politic_train = np.load(POLITIC_TRAIN_NPZ)
politic_test = np.load(POLITIC_TEST_NPZ)

# Load climate
climate_train = np.load(CLIMATE_TRAIN_NPZ)
climate_test = np.load(CLIMATE_TEST_NPZ)

print(f"   Politic train: {len(politic_train['y'])} rows")
print(f"   Politic test:  {len(politic_test['y'])} rows")
print(f"   Climate train: {len(climate_train['y'])} rows")
print(f"   Climate test:  {len(climate_test['y'])} rows")
print(f"   Subpop train:  {len(subpop_train_data['y'])} rows")
print(f"   Subpop test:   {len(subpop_test_data['y'])} rows")

# ============================================================================
# Create dataset_source arrays
# ============================================================================

print(f"\n6. Creating dataset_source arrays...")

politic_train_source = np.full(len(politic_train['y']), DATASET_SOURCE_POLITIC, dtype=np.int8)
politic_test_source = np.full(len(politic_test['y']), DATASET_SOURCE_POLITIC, dtype=np.int8)
climate_train_source = np.full(len(climate_train['y']), DATASET_SOURCE_CLIMATE, dtype=np.int8)
climate_test_source = np.full(len(climate_test['y']), DATASET_SOURCE_CLIMATE, dtype=np.int8)
subpop_train_source = np.full(len(subpop_train_data['y']), DATASET_SOURCE_SUBPOP, dtype=np.int8)
subpop_test_source = np.full(len(subpop_test_data['y']), DATASET_SOURCE_SUBPOP, dtype=np.int8)

# ============================================================================
# Concatenate all datasets
# ============================================================================

print(f"\n7. Concatenating all datasets...")

# Train
X_train = np.concatenate([
    politic_train['X'],
    climate_train['X'],
    subpop_train_data['X']
], axis=0).astype(np.float32)

y_train = np.concatenate([
    politic_train['y'],
    climate_train['y'],
    subpop_train_data['y']
], axis=0).astype(np.float32)

n_dims_train = np.concatenate([
    politic_train['n_dims'],
    climate_train['n_dims'],
    subpop_train_data['n_dims']
], axis=0).astype(np.int8)

qid_train = np.concatenate([
    politic_train['question_id'],
    climate_train['question_id'],
    subpop_train_data['question_id']
], axis=0).astype(np.int16)

cond_train = np.concatenate([
    politic_train['condition_code'],
    climate_train['condition_code'],
    subpop_train_data['condition_code']
], axis=0).astype(np.int8)

source_train = np.concatenate([
    politic_train_source,
    climate_train_source,
    subpop_train_source
], axis=0).astype(np.int8)

# Test
X_test = np.concatenate([
    politic_test['X'],
    climate_test['X'],
    subpop_test_data['X']
], axis=0).astype(np.float32)

y_test = np.concatenate([
    politic_test['y'],
    climate_test['y'],
    subpop_test_data['y']
], axis=0).astype(np.float32)

n_dims_test = np.concatenate([
    politic_test['n_dims'],
    climate_test['n_dims'],
    subpop_test_data['n_dims']
], axis=0).astype(np.int8)

qid_test = np.concatenate([
    politic_test['question_id'],
    climate_test['question_id'],
    subpop_test_data['question_id']
], axis=0).astype(np.int16)

cond_test = np.concatenate([
    politic_test['condition_code'],
    climate_test['condition_code'],
    subpop_test_data['condition_code']
], axis=0).astype(np.int8)

source_test = np.concatenate([
    politic_test_source,
    climate_test_source,
    subpop_test_source
], axis=0).astype(np.int8)

print(f"   Train: {X_train.shape} (politic + climate + subpop)")
print(f"   Test:  {X_test.shape} (politic + climate + subpop)")
print(f"   Train source counts: {np.bincount(source_train)}")
print(f"   Test source counts: {np.bincount(source_test)}")

# ============================================================================
# Save combined train/test
# ============================================================================

print(f"\n8. Saving combined train/test NPZ files...")

os.makedirs(OUT_DIR, exist_ok=True)

np.savez_compressed(
    OUT_TRAIN_NPZ,
    X=X_train, y=y_train, n_dims=n_dims_train, question_id=qid_train,
    condition_code=cond_train, dataset_source=source_train
)
combined_train_mb = os.path.getsize(OUT_TRAIN_NPZ) / 1024 / 1024
print(f"   Saved {OUT_TRAIN_NPZ} ({combined_train_mb:.1f} MB)")

np.savez_compressed(
    OUT_TEST_NPZ,
    X=X_test, y=y_test, n_dims=n_dims_test, question_id=qid_test,
    condition_code=cond_test, dataset_source=source_test
)
combined_test_mb = os.path.getsize(OUT_TEST_NPZ) / 1024 / 1024
print(f"   Saved {OUT_TEST_NPZ} ({combined_test_mb:.1f} MB)")

# ============================================================================
# Merge metadata
# ============================================================================

print(f"\n9. Merging metadata...")

# Load existing metadata
with open('train_test_split/probe_dataset_combined_train_metadata.json') as f:
    old_train_meta = json.load(f)
with open('train_test_split/probe_dataset_combined_test_metadata.json') as f:
    old_test_meta = json.load(f)

# Merge rows and create new metadata
def merge_metadata(old_meta, subpop_data, split_name, source_counts):
    """Merge old metadata with subpop metadata."""
    old_rows = old_meta['rows']
    subpop_rows = subpop_data['rows']

    # Offset subpop row indices
    for i, row in enumerate(subpop_rows):
        row['row_index'] = len(old_rows) + i
        row['dataset_source'] = 'subpop'

    merged_rows = old_rows + subpop_rows

    return {
        "set_name": split_name,
        "n_rows": {
            "total": len(merged_rows),
            "politic": old_meta['n_rows'].get('politic', old_meta['n_rows'].get('total', 0)),
            "climate": old_meta['n_rows'].get('climate', 0),
            "subpop": len(subpop_rows),
        },
        "n_respondents": {
            "total": old_meta['n_respondents'].get('total', 0) + len(subpop_rows),
            "politic": old_meta['n_respondents'].get('politic', 0),
            "climate": old_meta['n_respondents'].get('climate', 0),
            "subpop": len(subpop_rows),
        },
        "min_n": {
            "politic": old_meta['min_n'].get('politic', 3),
            "climate": old_meta['min_n'].get('climate', 2),
            "subpop": 1,
        },
        "dimensions": {
            "politic": old_meta['dimensions'].get('politic', []),
            "climate": old_meta['dimensions'].get('climate', []),
            "subpop": ["group"],
        },
        "condition_code_map": old_meta['condition_code_map'],
        "dataset_source_map": {
            "politic": 0,
            "climate": 1,
            "subpop": 2,
        },
        "n_dims_distribution": {
            "politic": old_meta['n_dims_distribution'].get('politic', {}),
            "climate": old_meta['n_dims_distribution'].get('climate', {}),
            "subpop": {"1": len(subpop_rows)},
        },
        "rows": merged_rows,
    }

train_counts = np.bincount(source_train)
test_counts = np.bincount(source_test)

combined_train_meta = merge_metadata(old_train_meta, subpop_train_data, "train", train_counts)
combined_test_meta = merge_metadata(old_test_meta, subpop_test_data, "test", test_counts)

with open(OUT_TRAIN_META, 'w') as f:
    json.dump(combined_train_meta, f, indent=2)
print(f"   Saved {OUT_TRAIN_META}")

with open(OUT_TEST_META, 'w') as f:
    json.dump(combined_test_meta, f, indent=2)
print(f"   Saved {OUT_TEST_META}")

# ============================================================================
# Summary
# ============================================================================

print(f"\n" + "="*80)
print("SUBPOP DATASET COMBINATION COMPLETE")
print("="*80)

print(f"\n📊 Dataset Summary:")
print(f"\n  Politic:")
print(f"    Train: {np.sum(source_train == 0):>8d} rows")
print(f"    Test:  {np.sum(source_test == 0):>8d} rows")

print(f"\n  Climate:")
print(f"    Train: {np.sum(source_train == 1):>8d} rows")
print(f"    Test:  {np.sum(source_test == 1):>8d} rows")

print(f"\n  Subpop:")
print(f"    Train: {np.sum(source_train == 2):>8d} rows ({100*np.sum(source_train==2)/len(source_train):.1f}%)")
print(f"    Test:  {np.sum(source_test == 2):>8d} rows ({100*np.sum(source_test==2)/len(source_test):.1f}%)")

print(f"\n  Combined:")
print(f"    Train: {len(X_train):>8d} rows total")
print(f"    Test:  {len(X_test):>8d} rows total")

print(f"\n✅ Ready for triple-source probe training!")
print(f"\n   Use: ridge_probe_model_triple.joblib")
print(f"   Input: {OUT_TRAIN_NPZ} + {OUT_TEST_NPZ}")
print(f"   Metadata: {OUT_TRAIN_META} + {OUT_TEST_META}")

print(f"\n⚠️  NOTE: Always condition on dataset_source (0=politic, 1=climate, 2=subpop)!")
print(f"   Each source has different semantics and dimensions.")
