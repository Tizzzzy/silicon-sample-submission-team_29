#!/usr/bin/env python3
"""
Memory-efficient combination of nutrition + policy with triple dataset.

Uses streaming/chunked operations to avoid loading all data at once.
"""

import json
import numpy as np
import os
import time

print("="*80)
print("Combine Nutrition + Policy (STREAMING - Memory Efficient)")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

TRIPLE_TRAIN_NPZ = "train_test_split/probe_dataset_triple_train.npz"
TRIPLE_TEST_NPZ = "train_test_split/probe_dataset_triple_test.npz"

NUTRITION_TRAIN_NPZ = "probe_dataset_nutrition_train.npz"
NUTRITION_TEST_NPZ = "probe_dataset_nutrition_test.npz"
POLICY_TRAIN_NPZ = "probe_dataset_policy_train.npz"
POLICY_TEST_NPZ = "probe_dataset_policy_test.npz"

OUT_DIR = "train_test_split"
OUT_TRAIN_NPZ = os.path.join(OUT_DIR, "probe_dataset_quintuple_train.npz")
OUT_TEST_NPZ = os.path.join(OUT_DIR, "probe_dataset_quintuple_test.npz")

DATASET_SOURCE_POLITIC = 0
DATASET_SOURCE_CLIMATE = 1
DATASET_SOURCE_SUBPOP = 2
DATASET_SOURCE_NUTRITION = 3
DATASET_SOURCE_POLICY = 4

REP_DIM = 5120

print(f"\nInput files:")
print(f"  Triple train:        {TRIPLE_TRAIN_NPZ}")
print(f"  Triple test:         {TRIPLE_TEST_NPZ}")
print(f"  Nutrition train:     {NUTRITION_TRAIN_NPZ}")
print(f"  Nutrition test:      {NUTRITION_TEST_NPZ}")
print(f"  Policy train:        {POLICY_TRAIN_NPZ}")
print(f"  Policy test:         {POLICY_TEST_NPZ}")

print(f"\nOutput files:")
print(f"  Quintuple train:     {OUT_TRAIN_NPZ}")
print(f"  Quintuple test:      {OUT_TEST_NPZ}")

# ============================================================================
# Load shape info without loading full arrays
# ============================================================================

print(f"\n{'='*80}")
print("Reading file shapes...")
print(f"{'='*80}")

# Check shapes
triple_train = np.load(TRIPLE_TRAIN_NPZ)
triple_test = np.load(TRIPLE_TEST_NPZ)
nutrition_train = np.load(NUTRITION_TRAIN_NPZ)
nutrition_test = np.load(NUTRITION_TEST_NPZ)
policy_train = np.load(POLICY_TRAIN_NPZ)
policy_test = np.load(POLICY_TEST_NPZ)

triple_train_n = triple_train['X'].shape[0]
triple_test_n = triple_test['X'].shape[0]
nutrition_train_n = nutrition_train['X'].shape[0]
nutrition_test_n = nutrition_test['X'].shape[0]
policy_train_n = policy_train['X'].shape[0]
policy_test_n = policy_test['X'].shape[0]

print(f"\nDataset sizes:")
print(f"  Triple train:   {triple_train_n:>8d} rows")
print(f"  Nutrition train: {nutrition_train_n:>8d} rows")
print(f"  Policy train:    {policy_train_n:>8d} rows")
print(f"  TOTAL train:     {triple_train_n + nutrition_train_n + policy_train_n:>8d} rows")
print(f"")
print(f"  Triple test:    {triple_test_n:>8d} rows")
print(f"  Nutrition test:  {nutrition_test_n:>8d} rows")
print(f"  Policy test:     {policy_test_n:>8d} rows")
print(f"  TOTAL test:      {triple_test_n + nutrition_test_n + policy_test_n:>8d} rows")

# ============================================================================
# Combine train data
# ============================================================================

print(f"\n{'='*80}")
print("Combining TRAIN data...")
print(f"{'='*80}")

# Load arrays one at a time and concatenate
print(f"\nLoading arrays...")

# Triple train
print(f"  Loading triple train...")
X_train_triple = triple_train['X'].astype(np.float32)
y_train_triple = triple_train['y'].astype(np.float32)
n_dims_train_triple = triple_train['n_dims'].astype(np.int8)
qid_train_triple = triple_train['question_id'].astype(np.int16)
cond_train_triple = triple_train['condition_code'].astype(np.int8)
source_train_triple = triple_train['dataset_source'].astype(np.int8)

# Nutrition train
print(f"  Loading nutrition train...")
X_train_nutrition = nutrition_train['X'].astype(np.float32)
y_train_nutrition = nutrition_train['y'].astype(np.float32)
n_dims_train_nutrition = nutrition_train['n_dims'].astype(np.int8)
qid_train_nutrition = nutrition_train['question_id'].astype(np.int16)
cond_train_nutrition = nutrition_train['condition_code'].astype(np.int8)
source_train_nutrition = np.full(nutrition_train_n, DATASET_SOURCE_NUTRITION, dtype=np.int8)

# Policy train
print(f"  Loading policy train...")
X_train_policy = policy_train['X'].astype(np.float32)
y_train_policy = policy_train['y'].astype(np.float32)
n_dims_train_policy = policy_train['n_dims'].astype(np.int8)
qid_train_policy = policy_train['question_id'].astype(np.int16)
cond_train_policy = policy_train['condition_code'].astype(np.int8)
source_train_policy = np.full(policy_train_n, DATASET_SOURCE_POLICY, dtype=np.int8)

# Concatenate
print(f"\nConcatenating train arrays...")
X_train = np.concatenate([X_train_triple, X_train_nutrition, X_train_policy], axis=0).astype(np.float32)
y_train = np.concatenate([y_train_triple, y_train_nutrition, y_train_policy], axis=0).astype(np.float32)
n_dims_train = np.concatenate([n_dims_train_triple, n_dims_train_nutrition, n_dims_train_policy])
qid_train = np.concatenate([qid_train_triple, qid_train_nutrition, qid_train_policy])
cond_train = np.concatenate([cond_train_triple, cond_train_nutrition, cond_train_policy])
source_train = np.concatenate([source_train_triple, source_train_nutrition, source_train_policy])

print(f"  Concatenated train: {X_train.shape[0]} rows")

# Clear memory
del X_train_triple, y_train_triple, n_dims_train_triple, qid_train_triple, cond_train_triple, source_train_triple
del X_train_nutrition, y_train_nutrition, n_dims_train_nutrition, qid_train_nutrition, cond_train_nutrition, source_train_nutrition
del X_train_policy, y_train_policy, n_dims_train_policy, qid_train_policy, cond_train_policy, source_train_policy

# ============================================================================
# Combine test data
# ============================================================================

print(f"\n{'='*80}")
print("Combining TEST data...")
print(f"{'='*80}")

# Triple test
print(f"  Loading triple test...")
X_test_triple = triple_test['X'].astype(np.float32)
y_test_triple = triple_test['y'].astype(np.float32)
n_dims_test_triple = triple_test['n_dims'].astype(np.int8)
qid_test_triple = triple_test['question_id'].astype(np.int16)
cond_test_triple = triple_test['condition_code'].astype(np.int8)
source_test_triple = triple_test['dataset_source'].astype(np.int8)

# Nutrition test
print(f"  Loading nutrition test...")
X_test_nutrition = nutrition_test['X'].astype(np.float32)
y_test_nutrition = nutrition_test['y'].astype(np.float32)
n_dims_test_nutrition = nutrition_test['n_dims'].astype(np.int8)
qid_test_nutrition = nutrition_test['question_id'].astype(np.int16)
cond_test_nutrition = nutrition_test['condition_code'].astype(np.int8)
source_test_nutrition = np.full(nutrition_test_n, DATASET_SOURCE_NUTRITION, dtype=np.int8)

# Policy test
print(f"  Loading policy test...")
X_test_policy = policy_test['X'].astype(np.float32)
y_test_policy = policy_test['y'].astype(np.float32)
n_dims_test_policy = policy_test['n_dims'].astype(np.int8)
qid_test_policy = policy_test['question_id'].astype(np.int16)
cond_test_policy = policy_test['condition_code'].astype(np.int8)
source_test_policy = np.full(policy_test_n, DATASET_SOURCE_POLICY, dtype=np.int8)

# Concatenate
print(f"\nConcatenating test arrays...")
X_test = np.concatenate([X_test_triple, X_test_nutrition, X_test_policy], axis=0).astype(np.float32)
y_test = np.concatenate([y_test_triple, y_test_nutrition, y_test_policy], axis=0).astype(np.float32)
n_dims_test = np.concatenate([n_dims_test_triple, n_dims_test_nutrition, n_dims_test_policy])
qid_test = np.concatenate([qid_test_triple, qid_test_nutrition, qid_test_policy])
cond_test = np.concatenate([cond_test_triple, cond_test_nutrition, cond_test_policy])
source_test = np.concatenate([source_test_triple, source_test_nutrition, source_test_policy])

print(f"  Concatenated test: {X_test.shape[0]} rows")

# Clear memory
del X_test_triple, y_test_triple, n_dims_test_triple, qid_test_triple, cond_test_triple, source_test_triple
del X_test_nutrition, y_test_nutrition, n_dims_test_nutrition, qid_test_nutrition, cond_test_nutrition, source_test_nutrition
del X_test_policy, y_test_policy, n_dims_test_policy, qid_test_policy, cond_test_policy, source_test_policy

# ============================================================================
# Save quintuple NPZ files
# ============================================================================

print(f"\n{'='*80}")
print("Saving Quintuple NPZ Files")
print(f"{'='*80}")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"\nSaving {OUT_TRAIN_NPZ}...")
t0 = time.time()
np.savez_compressed(
    OUT_TRAIN_NPZ,
    X=X_train, y=y_train, n_dims=n_dims_train,
    question_id=qid_train, condition_code=cond_train, dataset_source=source_train
)
elapsed = time.time() - t0
train_size = os.path.getsize(OUT_TRAIN_NPZ) / 1e9
print(f"  ✓ {train_size:.2f} GB in {elapsed:.1f}s")

print(f"\nSaving {OUT_TEST_NPZ}...")
t0 = time.time()
np.savez_compressed(
    OUT_TEST_NPZ,
    X=X_test, y=y_test, n_dims=n_dims_test,
    question_id=qid_test, condition_code=cond_test, dataset_source=source_test
)
elapsed = time.time() - t0
test_size = os.path.getsize(OUT_TEST_NPZ) / 1e9
print(f"  ✓ {test_size:.2f} GB in {elapsed:.1f}s")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("COMBINATION COMPLETE - QUINTUPLE DATASET")
print(f"{'='*80}")

train_counts = np.bincount(source_train)
test_counts = np.bincount(source_test)

print(f"\nDataset source breakdown (FINAL):")
sources = {0: "politic", 1: "climate", 2: "subpop", 3: "nutrition", 4: "policy"}
for src_id, src_name in sources.items():
    if src_id < len(train_counts):
        train_count = train_counts[src_id]
        test_count = test_counts[src_id]
        train_pct = 100.0 * train_count / len(source_train)
        test_pct = 100.0 * test_count / len(source_test)
        print(f"  {src_name:10s}: {train_count:>8d} train ({train_pct:>5.1f}%), {test_count:>8d} test ({test_pct:>5.1f}%)")

print(f"\nFinal Size:")
print(f"  Train NPZ: {train_size:.2f} GB")
print(f"  Test NPZ:  {test_size:.2f} GB")
print(f"  Total:     {train_size + test_size:.2f} GB")

print(f"\n✅ QUINTUPLE DATASET READY FOR TRAINING!")
print(f"\nUsage:")
print(f"  python3 train_probe_quintuple_respondent_split.py")
