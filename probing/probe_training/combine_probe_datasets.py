#!/usr/bin/env python3
"""
Combine politic and climate probe datasets into unified train/test sets.

CRITICAL: downstream consumers MUST condition on (dataset_source, question_id) jointly,
NOT on question_id alone. The two sources have overlapping question_id ranges that
mean different things:
  - politic: question_id is a real survey question index (0-19+)
  - climate: question_id is task_type index (0=happening, 1=cause, 2=worry, 3=priority)

The "dataset_source" array disambiguates them: use it.

Workflow:
1. Load politic train/test NPZ + metadata (unchanged, already built)
2. Load climate train/test NPZ + metadata (built by build_probe_dataset_climate_respondent_split.py)
3. Build unified condition_code_map (politic 4 codes + climate "N/A" as code 4)
4. Add dataset_source array (int8: 0=politic, 1=climate) to all rows
5. Concatenate arrays (politic first, then climate) for train and test separately
6. Merge metadata rows, tag each with "dataset_source" string label
7. Create nested metadata schema (per-source dimensions, min_n, n_dims_distribution, etc.)
8. Save to train_test_split/ directory
"""

import json
import numpy as np
import os
import time

# Configuration
POLITIC_TRAIN_NPZ = "probe_dataset_train.npz"
POLITIC_TEST_NPZ = "probe_dataset_test.npz"
POLITIC_TRAIN_META = "probe_dataset_train_metadata.json"
POLITIC_TEST_META = "probe_dataset_test_metadata.json"

CLIMATE_TRAIN_NPZ = "probe_dataset_climate_train.npz"
CLIMATE_TEST_NPZ = "probe_dataset_climate_test.npz"
CLIMATE_TRAIN_META = "probe_dataset_climate_train_metadata.json"
CLIMATE_TEST_META = "probe_dataset_climate_test_metadata.json"

OUT_DIR = "train_test_split"
OUT_TRAIN_NPZ = os.path.join(OUT_DIR, "probe_dataset_combined_train.npz")
OUT_TEST_NPZ = os.path.join(OUT_DIR, "probe_dataset_combined_test.npz")
OUT_TRAIN_META = os.path.join(OUT_DIR, "probe_dataset_combined_train_metadata.json")
OUT_TEST_META = os.path.join(OUT_DIR, "probe_dataset_combined_test_metadata.json")

DATASET_SOURCE_POLITIC = 0
DATASET_SOURCE_CLIMATE = 1

print("="*80)
print("Combine Politic and Climate Probe Datasets")
print("="*80)
print(f"\nConfiguration:")
print(f"  Politic train: {POLITIC_TRAIN_NPZ}")
print(f"  Politic test:  {POLITIC_TEST_NPZ}")
print(f"  Climate train: {CLIMATE_TRAIN_NPZ}")
print(f"  Climate test:  {CLIMATE_TEST_NPZ}")
print(f"  Output dir: {OUT_DIR}")

# ============================================================================
# Load inputs
# ============================================================================

print(f"\nLoading all datasets...")
t0 = time.time()

def load_npz_and_meta(npz_path, meta_path):
    """Load NPZ file and metadata JSON."""
    d = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return d, meta

politic_train_npz, politic_train_meta = load_npz_and_meta(POLITIC_TRAIN_NPZ, POLITIC_TRAIN_META)
politic_test_npz, politic_test_meta = load_npz_and_meta(POLITIC_TEST_NPZ, POLITIC_TEST_META)
climate_train_npz, climate_train_meta = load_npz_and_meta(CLIMATE_TRAIN_NPZ, CLIMATE_TRAIN_META)
climate_test_npz, climate_test_meta = load_npz_and_meta(CLIMATE_TEST_NPZ, CLIMATE_TEST_META)

print(f"Loaded all datasets in {time.time() - t0:.1f}s")
print(f"  Politic train: {len(politic_train_npz['y'])} rows")
print(f"  Politic test:  {len(politic_test_npz['y'])} rows")
print(f"  Climate train: {len(climate_train_npz['y'])} rows")
print(f"  Climate test:  {len(climate_test_npz['y'])} rows")

# ============================================================================
# Build unified condition_code_map
# ============================================================================

print(f"\nBuilding unified condition_code_map...")

# Verify train/test agreement within each source
assert politic_train_meta["condition_code_map"] == politic_test_meta["condition_code_map"], \
    "Politic train/test condition_code_maps differ!"
assert climate_train_meta["condition_code_map"] == climate_test_meta["condition_code_map"], \
    "Climate train/test condition_code_maps differ!"

def build_unified_condition_map(politic_map, climate_map):
    """Build unified map, keeping politic codes unchanged."""
    unified = dict(politic_map)
    next_code = max(unified.values()) + 1
    for label in climate_map:
        if label not in unified:
            unified[label] = next_code
            next_code += 1
    return unified

unified_condition_map = build_unified_condition_map(
    politic_train_meta["condition_code_map"],
    climate_train_meta["condition_code_map"]
)
print(f"  Unified condition_code_map: {unified_condition_map}")

# Remap climate condition codes
def remap_condition_codes(old_codes, old_map, unified_map):
    """Remap condition codes from old map to unified map."""
    code_to_label = {v: k for k, v in old_map.items()}
    old_to_unified = np.array(
        [unified_map[code_to_label[c]] for c in range(len(old_map))], dtype=np.int8
    )
    return old_to_unified[old_codes]

climate_train_cond_remapped = remap_condition_codes(
    climate_train_npz["condition_code"],
    climate_train_meta["condition_code_map"],
    unified_condition_map
)
climate_test_cond_remapped = remap_condition_codes(
    climate_test_npz["condition_code"],
    climate_test_meta["condition_code_map"],
    unified_condition_map
)

print(f"  Remapped climate condition codes: {np.unique(climate_train_cond_remapped)}")

# ============================================================================
# Create dataset_source arrays
# ============================================================================

print(f"\nCreating dataset_source arrays...")

politic_train_source = np.full(len(politic_train_npz["y"]), DATASET_SOURCE_POLITIC, dtype=np.int8)
politic_test_source = np.full(len(politic_test_npz["y"]), DATASET_SOURCE_POLITIC, dtype=np.int8)
climate_train_source = np.full(len(climate_train_npz["y"]), DATASET_SOURCE_CLIMATE, dtype=np.int8)
climate_test_source = np.full(len(climate_test_npz["y"]), DATASET_SOURCE_CLIMATE, dtype=np.int8)

print(f"  Politic sources: {np.bincount(politic_train_source)}")
print(f"  Climate sources: {np.bincount(climate_train_source)}")

# ============================================================================
# Concatenate arrays
# ============================================================================

print(f"\nConcatenating arrays...")

def concat_split(politic_npz, climate_npz, politic_source, climate_source,
                 climate_cond_remapped):
    """Concatenate arrays for train or test split."""
    X = np.concatenate([politic_npz["X"], climate_npz["X"]], axis=0).astype(np.float32)
    y = np.concatenate([politic_npz["y"], climate_npz["y"]], axis=0).astype(np.float32)
    n_dims = np.concatenate([politic_npz["n_dims"], climate_npz["n_dims"]], axis=0).astype(np.int8)
    question_id = np.concatenate([politic_npz["question_id"], climate_npz["question_id"]], axis=0).astype(np.int16)
    condition_code = np.concatenate([politic_npz["condition_code"], climate_cond_remapped], axis=0).astype(np.int8)
    dataset_source = np.concatenate([politic_source, climate_source], axis=0).astype(np.int8)
    return X, y, n_dims, question_id, condition_code, dataset_source

X_train, y_train, n_dims_train, qid_train, cond_train, source_train = concat_split(
    politic_train_npz, climate_train_npz, politic_train_source, climate_train_source,
    climate_train_cond_remapped
)
X_test, y_test, n_dims_test, qid_test, cond_test, source_test = concat_split(
    politic_test_npz, climate_test_npz, politic_test_source, climate_test_source,
    climate_test_cond_remapped
)

# Sanity checks
assert len(X_train) == len(politic_train_npz["X"]) + len(climate_train_npz["X"]), "Train X length mismatch!"
assert len(X_test) == len(politic_test_npz["X"]) + len(climate_test_npz["X"]), "Test X length mismatch!"
assert len(X_train) == len(y_train) == len(n_dims_train) == len(qid_train) == len(cond_train) == len(source_train), \
    "Train array length mismatch!"
assert len(X_test) == len(y_test) == len(n_dims_test) == len(qid_test) == len(cond_test) == len(source_test), \
    "Test array length mismatch!"

print(f"  Train shape: X={X_train.shape}, y={y_train.shape}, source={source_train.shape}")
print(f"  Test shape: X={X_test.shape}, y={y_test.shape}, source={source_test.shape}")
print(f"  Train source counts: {np.bincount(source_train)}")
print(f"  Test source counts: {np.bincount(source_test)}")

# ============================================================================
# Save combined NPZ
# ============================================================================

print(f"\nSaving combined NPZ files...")
os.makedirs(OUT_DIR, exist_ok=True)

np.savez_compressed(
    OUT_TRAIN_NPZ,
    X=X_train, y=y_train, n_dims=n_dims_train, question_id=qid_train,
    condition_code=cond_train, dataset_source=source_train
)
train_size_mb = os.path.getsize(OUT_TRAIN_NPZ) / 1024 / 1024
print(f"  Saved {len(X_train)} rows to {OUT_TRAIN_NPZ} ({train_size_mb:.1f} MB)")

np.savez_compressed(
    OUT_TEST_NPZ,
    X=X_test, y=y_test, n_dims=n_dims_test, question_id=qid_test,
    condition_code=cond_test, dataset_source=source_test
)
test_size_mb = os.path.getsize(OUT_TEST_NPZ) / 1024 / 1024
print(f"  Saved {len(X_test)} rows to {OUT_TEST_NPZ} ({test_size_mb:.1f} MB)")

# ============================================================================
# Merge metadata rows
# ============================================================================

print(f"\nMerging metadata rows...")

def merge_rows(politic_meta, climate_meta):
    """Merge rows from both sources, tagging each with dataset_source."""
    politic_rows = politic_meta["rows"]
    climate_rows = climate_meta["rows"]
    offset = len(politic_rows)

    merged = []
    for r in politic_rows:
        r2 = dict(r)
        r2["dataset_source"] = "politic"
        merged.append(r2)

    for r in climate_rows:
        r2 = dict(r)
        r2["row_index"] = r["row_index"] + offset
        r2["dataset_source"] = "climate"
        merged.append(r2)

    return merged

merged_train_rows = merge_rows(politic_train_meta, climate_train_meta)
merged_test_rows = merge_rows(politic_test_meta, climate_test_meta)

print(f"  Merged train rows: {len(merged_train_rows)}")
print(f"  Merged test rows: {len(merged_test_rows)}")

# ============================================================================
# Build combined metadata
# ============================================================================

print(f"\nBuilding combined metadata...")

def build_combined_metadata(set_name, politic_meta, climate_meta, merged_rows, unified_condition_map):
    """Build combined metadata schema."""
    return {
        "set_name": set_name,
        "n_rows": {
            "total": len(merged_rows),
            "politic": politic_meta["n_rows"],
            "climate": climate_meta["n_rows"],
        },
        "n_respondents": {
            "total": politic_meta["n_respondents"] + climate_meta["n_respondents"],
            "politic": politic_meta["n_respondents"],
            "climate": climate_meta["n_respondents"],
        },
        "min_n": {
            "politic": politic_meta["min_n"],
            "climate": climate_meta["min_n"],
        },
        "dimensions": {
            "politic": politic_meta["dimensions"],
            "climate": climate_meta["dimensions"],
        },
        "condition_code_map": unified_condition_map,
        "dataset_source_map": {
            "politic": DATASET_SOURCE_POLITIC,
            "climate": DATASET_SOURCE_CLIMATE,
        },
        "n_dims_distribution": {
            "politic": politic_meta["n_dims_distribution"],
            "climate": climate_meta["n_dims_distribution"],
        },
        "rows": merged_rows,
    }

combined_train_meta = build_combined_metadata(
    "train", politic_train_meta, climate_train_meta, merged_train_rows, unified_condition_map
)
combined_test_meta = build_combined_metadata(
    "test", politic_test_meta, climate_test_meta, merged_test_rows, unified_condition_map
)

# ============================================================================
# Save metadata
# ============================================================================

print(f"\nSaving combined metadata...")

with open(OUT_TRAIN_META, 'w') as f:
    json.dump(combined_train_meta, f, indent=2)
print(f"  Saved {OUT_TRAIN_META}")

with open(OUT_TEST_META, 'w') as f:
    json.dump(combined_test_meta, f, indent=2)
print(f"  Saved {OUT_TEST_META}")

# ============================================================================
# Final summary
# ============================================================================

print(f"\n" + "="*80)
print("Combined Dataset Summary")
print("="*80)

for set_name, npz_len, meta in [("Train", len(X_train), combined_train_meta),
                                 ("Test", len(X_test), combined_test_meta)]:
    print(f"\n{set_name} set:")
    print(f"  Total rows: {meta['n_rows']['total']}")
    print(f"    Politic: {meta['n_rows']['politic']} rows")
    print(f"    Climate: {meta['n_rows']['climate']} rows")

    print(f"  Total respondents: {meta['n_respondents']['total']}")
    print(f"    Politic: {meta['n_respondents']['politic']}")
    print(f"    Climate: {meta['n_respondents']['climate']}")

    print(f"  MIN_N:")
    print(f"    Politic: {meta['min_n']['politic']}")
    print(f"    Climate: {meta['min_n']['climate']}")

    print(f"  Condition distribution:")
    for cond, code in meta['condition_code_map'].items():
        print(f"    {cond}: code {code}")

print(f"\n✅ Combined datasets ready for training!")
print(f"   Train: {OUT_TRAIN_NPZ}")
print(f"   Test: {OUT_TEST_NPZ}")
print(f"\n⚠️  IMPORTANT: Always condition on (dataset_source, question_id) jointly!")
print(f"   question_id ranges overlap between sources but mean different things.")
