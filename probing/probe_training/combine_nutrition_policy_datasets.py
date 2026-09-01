#!/usr/bin/env python3
"""
Combine nutrition and policy datasets with existing triple dataset (politic + climate + subpop).

Workflow:
1. Load nutrition dataset (individual-level, needs grouping)
   - Split 80/20 at respondent level
   - Group by: question_type + experimental_text (condition) + demographic_dimensions
   - Average representations and opinions
2. Load policy dataset (individual-level, needs grouping)
   - Split 80/20 at respondent level
   - Group by: question_type + condition + demographic_dimensions
   - Average representations and opinions
3. Combine both with existing triple dataset (politic + climate + subpop)
4. Output: 5-source train/test NPZ files

This script creates a QUINTUPLE-SOURCE dataset for comprehensive opinion probing.
"""

import json
import numpy as np
import os
import time
from collections import defaultdict
from datetime import datetime

print("="*80)
print("Combine Nutrition + Policy with Triple Dataset (Politic + Climate + Subpop)")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

NUTRITION_INPUT = "../representation_extract/extracted_representations_nutrition.json"
POLICY_INPUT = "../representation_extract/extracted_representations_policy.json"

# Existing triple dataset files
TRIPLE_TRAIN_NPZ = "train_test_split/probe_dataset_triple_train.npz"
TRIPLE_TEST_NPZ = "train_test_split/probe_dataset_triple_test.npz"

# Output files
NUTRITION_OUTPUT_TRAIN_NPZ = "probe_dataset_nutrition_train.npz"
NUTRITION_OUTPUT_TEST_NPZ = "probe_dataset_nutrition_test.npz"
POLICY_OUTPUT_TRAIN_NPZ = "probe_dataset_policy_train.npz"
POLICY_OUTPUT_TEST_NPZ = "probe_dataset_policy_test.npz"

OUT_DIR = "train_test_split"
OUT_TRAIN_NPZ = os.path.join(OUT_DIR, "probe_dataset_quintuple_train.npz")
OUT_TEST_NPZ = os.path.join(OUT_DIR, "probe_dataset_quintuple_test.npz")
OUT_TRAIN_META = os.path.join(OUT_DIR, "probe_dataset_quintuple_train_metadata.json")
OUT_TEST_META = os.path.join(OUT_DIR, "probe_dataset_quintuple_test_metadata.json")

DATASET_SOURCE_POLITIC = 0
DATASET_SOURCE_CLIMATE = 1
DATASET_SOURCE_SUBPOP = 2
DATASET_SOURCE_NUTRITION = 3
DATASET_SOURCE_POLICY = 4

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
REP_DIM = 5120

print(f"\nDataset sources:")
print(f"  0 = politic")
print(f"  1 = climate")
print(f"  2 = subpop")
print(f"  3 = nutrition (NEW)")
print(f"  4 = policy (NEW)")

# ============================================================================
# Helper function to build individual-level dataset
# ============================================================================

def build_grouped_dataset(input_file, dataset_name, text_type_field, question_type_field,
                         respondent_id_field, opinion_field, rep_field='last_token_residual_stream'):
    """
    Load individual-level data, split respondents 80/20, group, and build NPZ.

    Args:
        input_file: Path to JSON file
        dataset_name: Name for logging
        text_type_field: Key name for text condition (e.g., 'experimental_text' or 'condition')
        question_type_field: Key name for question type (e.g., 'task_type' or 'target_variable')
        respondent_id_field: Key name for respondent ID (e.g., 'case_ID' or 'case_id')
        opinion_field: Key name for opinion value (e.g., 'target' or 'ground_truth')
        rep_field: Key name for representation vector (default: 'last_token_residual_stream')
    """
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name.upper()}")
    print(f"{'='*80}")

    # ========================================================================
    # Load data
    # ========================================================================

    print(f"\n1. Loading {input_file}...")
    t0 = time.time()

    with open(input_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'representations' in data:
        entries = data['representations']
    else:
        entries = data

    print(f"   Loaded {len(entries)} entries in {time.time()-t0:.1f}s")

    if len(entries) == 0:
        raise ValueError(f"{dataset_name} data is empty!")

    sample = entries[0]
    print(f"   Sample keys: {list(sample.keys())[:10]}...")

    # ========================================================================
    # Split respondents 80/20
    # ========================================================================

    print(f"\n2. Splitting respondents 80/20...")

    # Use specified respondent field
    if respondent_id_field not in sample:
        raise ValueError(f"Cannot find respondent identifier field '{respondent_id_field}' in {dataset_name}")

    respondent_key = respondent_id_field

    # Convert to int, handling string/float representations
    respondent_ids = sorted(set(int(float(str(e[respondent_key]))) for e in entries))
    print(f"   Unique respondents: {len(respondent_ids)}")

    np.random.seed(RANDOM_SEED)
    n_train = int(len(respondent_ids) * TRAIN_RATIO)
    train_ids = set(np.random.choice(respondent_ids, n_train, replace=False))
    test_ids = set(respondent_ids) - train_ids

    train_entries = [e for e in entries if int(float(str(e[respondent_key]))) in train_ids]
    test_entries = [e for e in entries if int(float(str(e[respondent_key]))) in test_ids]

    print(f"   Train: {len(train_entries)} entries ({len(train_ids)} respondents)")
    print(f"   Test:  {len(test_entries)} entries ({len(test_ids)} respondents)")

    # ========================================================================
    # Group and aggregate
    # ========================================================================

    print(f"\n3. Grouping and aggregating...")

    def aggregate_split(split_entries, split_name):
        """Aggregate entries by group."""
        groups = defaultdict(lambda: {
            'sum_vec': np.zeros(REP_DIM, dtype=np.float64),
            'sum_opinion': 0.0,
            'n': 0,
            'question_label': None,
        })

        for entry in split_entries:
            # Get representation
            if rep_field in entry:
                vec = np.asarray(entry[rep_field], dtype=np.float64)
            elif 'representation' in entry:
                vec = np.asarray(entry['representation'], dtype=np.float64)
            else:
                raise ValueError(f"Cannot find representation in {dataset_name} (tried '{rep_field}')")

            # Get opinion using specified field
            if opinion_field not in entry:
                raise ValueError(f"Cannot find opinion field '{opinion_field}' in {dataset_name}")
            opinion = float(entry[opinion_field])

            # Build group key
            text_type = str(entry.get(text_type_field, 'unknown'))
            question_type = str(entry.get(question_type_field, 'unknown'))

            # Get demographics (variable by dataset)
            exclude_fields = {'representation', 'last_token_residual_stream', rep_field,
                             respondent_key, text_type_field, question_type_field,
                             'individual_opinion', 'target', 'ground_truth', opinion_field,
                             'group_id', 'respondent_id', 'response_id', 'prompt',
                             'input', 'representation_shape', 'task_type', 'target_variable',
                             'condition', 'experimental_text', 'case_ID', 'case_id'}
            demo_dict = {k: v for k, v in entry.items()
                        if k not in exclude_fields}
            demo_tuple = tuple(sorted(demo_dict.items()))

            group_key = (question_type, text_type, demo_tuple)

            g = groups[group_key]
            g['sum_vec'] += vec
            g['sum_opinion'] += opinion
            g['n'] += 1
            g['question_label'] = entry.get('question_label', question_type)

        print(f"   {split_name}: {len(groups)} groups from {len(split_entries)} entries")
        return groups

    train_groups = aggregate_split(train_entries, "Train")
    test_groups = aggregate_split(test_entries, "Test")

    # ========================================================================
    # Build rows and arrays
    # ========================================================================

    print(f"\n4. Building NPZ arrays...")

    def build_arrays(groups):
        """Convert groups to NPZ arrays."""
        rows = []
        X_list = []
        y_list = []

        for idx, ((qtype, text_type, demo_tuple), g) in enumerate(groups.items()):
            mean_vec = (g['sum_vec'] / g['n']).astype(np.float32)
            mean_opinion = g['sum_opinion'] / g['n']

            X_list.append(mean_vec)
            y_list.append(mean_opinion)

            # Reconstruct dimension dict
            demo_dict = dict(demo_tuple)

            rows.append({
                'row_index': idx,
                'n': g['n'],
                'n_dims': len(demo_dict),
                'dimension_subset': list(demo_dict.keys()),
                'dimension_values': demo_dict,
                'condition': text_type,
                'question_id': hash(qtype) % 1000,
                'question_label': g['question_label'],
                'y_mean': mean_opinion,
            })

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        n_dims = np.array([r['n_dims'] for r in rows], dtype=np.int8)
        question_id = np.array([r['question_id'] for r in rows], dtype=np.int16)
        condition_code = np.zeros(len(y), dtype=np.int8)  # All N/A for these datasets

        return X, y, n_dims, question_id, condition_code, rows

    X_train, y_train, n_dims_train, qid_train, cond_train, rows_train = build_arrays(train_groups)
    X_test, y_test, n_dims_test, qid_test, cond_test, rows_test = build_arrays(test_groups)

    print(f"   Train: {X_train.shape[0]} rows")
    print(f"   Test:  {X_test.shape[0]} rows")

    return {
        'X_train': X_train, 'y_train': y_train, 'n_dims_train': n_dims_train,
        'qid_train': qid_train, 'cond_train': cond_train, 'rows_train': rows_train,
        'X_test': X_test, 'y_test': y_test, 'n_dims_test': n_dims_test,
        'qid_test': qid_test, 'cond_test': cond_test, 'rows_test': rows_test,
        'n_train_respondents': len(train_ids), 'n_test_respondents': len(test_ids),
    }

# ========================================================================
# Build nutrition dataset
# ========================================================================

nutrition_data = build_grouped_dataset(
    NUTRITION_INPUT, "nutrition",
    text_type_field='experimental_text',
    question_type_field='task_type',
    respondent_id_field='case_ID',
    opinion_field='target'
)

# ========================================================================
# Build policy dataset
# ========================================================================

policy_data = build_grouped_dataset(
    POLICY_INPUT, "policy",
    text_type_field='condition',
    question_type_field='target_variable',
    respondent_id_field='case_id',
    opinion_field='ground_truth'
)

# ========================================================================
# Save nutrition and policy NPZ files
# ========================================================================

print(f"\n{'='*80}")
print("Saving nutrition and policy NPZ files...")
print(f"{'='*80}")

os.makedirs(OUT_DIR, exist_ok=True)

# Nutrition
np.savez_compressed(
    NUTRITION_OUTPUT_TRAIN_NPZ,
    X=nutrition_data['X_train'], y=nutrition_data['y_train'],
    n_dims=nutrition_data['n_dims_train'], question_id=nutrition_data['qid_train'],
    condition_code=nutrition_data['cond_train'],
)
print(f"✓ {NUTRITION_OUTPUT_TRAIN_NPZ}")

np.savez_compressed(
    NUTRITION_OUTPUT_TEST_NPZ,
    X=nutrition_data['X_test'], y=nutrition_data['y_test'],
    n_dims=nutrition_data['n_dims_test'], question_id=nutrition_data['qid_test'],
    condition_code=nutrition_data['cond_test'],
)
print(f"✓ {NUTRITION_OUTPUT_TEST_NPZ}")

# Policy
np.savez_compressed(
    POLICY_OUTPUT_TRAIN_NPZ,
    X=policy_data['X_train'], y=policy_data['y_train'],
    n_dims=policy_data['n_dims_train'], question_id=policy_data['qid_train'],
    condition_code=policy_data['cond_train'],
)
print(f"✓ {POLICY_OUTPUT_TRAIN_NPZ}")

np.savez_compressed(
    POLICY_OUTPUT_TEST_NPZ,
    X=policy_data['X_test'], y=policy_data['y_test'],
    n_dims=policy_data['n_dims_test'], question_id=policy_data['qid_test'],
    condition_code=policy_data['cond_test'],
)
print(f"✓ {POLICY_OUTPUT_TEST_NPZ}")

# ========================================================================
# Load and combine with triple dataset
# ========================================================================

print(f"\n{'='*80}")
print("Loading and combining with triple dataset...")
print(f"{'='*80}")

triple_train = np.load(TRIPLE_TRAIN_NPZ)
triple_test = np.load(TRIPLE_TEST_NPZ)

print(f"\nExisting triple dataset:")
print(f"  Train: {len(triple_train['y'])} rows")
print(f"  Test:  {len(triple_test['y'])} rows")

print(f"\nNew datasets:")
print(f"  Nutrition train: {len(nutrition_data['y_train'])} rows")
print(f"  Nutrition test:  {len(nutrition_data['y_test'])} rows")
print(f"  Policy train:    {len(policy_data['y_train'])} rows")
print(f"  Policy test:     {len(policy_data['y_test'])} rows")

# Create source arrays
triple_train_source = triple_train['dataset_source']
triple_test_source = triple_test['dataset_source']
nutrition_train_source = np.full(len(nutrition_data['y_train']), DATASET_SOURCE_NUTRITION, dtype=np.int8)
nutrition_test_source = np.full(len(nutrition_data['y_test']), DATASET_SOURCE_NUTRITION, dtype=np.int8)
policy_train_source = np.full(len(policy_data['y_train']), DATASET_SOURCE_POLICY, dtype=np.int8)
policy_test_source = np.full(len(policy_data['y_test']), DATASET_SOURCE_POLICY, dtype=np.int8)

# Concatenate all
print(f"\nConcatenating all sources...")

X_train_quintuple = np.concatenate([
    triple_train['X'], nutrition_data['X_train'], policy_data['X_train']
], axis=0).astype(np.float32)

y_train_quintuple = np.concatenate([
    triple_train['y'], nutrition_data['y_train'], policy_data['y_train']
], axis=0).astype(np.float32)

source_train_quintuple = np.concatenate([
    triple_train_source, nutrition_train_source, policy_train_source
], axis=0).astype(np.int8)

X_test_quintuple = np.concatenate([
    triple_test['X'], nutrition_data['X_test'], policy_data['X_test']
], axis=0).astype(np.float32)

y_test_quintuple = np.concatenate([
    triple_test['y'], nutrition_data['y_test'], policy_data['y_test']
], axis=0).astype(np.float32)

source_test_quintuple = np.concatenate([
    triple_test_source, nutrition_test_source, policy_test_source
], axis=0).astype(np.int8)

print(f"  Train: {X_train_quintuple.shape[0]} total rows")
print(f"  Test:  {X_test_quintuple.shape[0]} total rows")

# Also concatenate other arrays
n_dims_train = np.concatenate([triple_train['n_dims'], nutrition_data['n_dims_train'], policy_data['n_dims_train']])
n_dims_test = np.concatenate([triple_test['n_dims'], nutrition_data['n_dims_test'], policy_data['n_dims_test']])
qid_train = np.concatenate([triple_train['question_id'], nutrition_data['qid_train'], policy_data['qid_train']])
qid_test = np.concatenate([triple_test['question_id'], nutrition_data['qid_test'], policy_data['qid_test']])
cond_train = np.concatenate([triple_train['condition_code'], nutrition_data['cond_train'], policy_data['cond_train']])
cond_test = np.concatenate([triple_test['condition_code'], nutrition_data['cond_test'], policy_data['cond_test']])

# ========================================================================
# Save quintuple dataset
# ========================================================================

print(f"\nSaving quintuple NPZ files...")

np.savez_compressed(
    OUT_TRAIN_NPZ,
    X=X_train_quintuple, y=y_train_quintuple, n_dims=n_dims_train,
    question_id=qid_train, condition_code=cond_train, dataset_source=source_train_quintuple
)
print(f"✓ {OUT_TRAIN_NPZ}")

np.savez_compressed(
    OUT_TEST_NPZ,
    X=X_test_quintuple, y=y_test_quintuple, n_dims=n_dims_test,
    question_id=qid_test, condition_code=cond_test, dataset_source=source_test_quintuple
)
print(f"✓ {OUT_TEST_NPZ}")

# ========================================================================
# Summary
# ========================================================================

print(f"\n{'='*80}")
print("COMBINATION COMPLETE - QUINTUPLE SOURCE DATASET")
print(f"{'='*80}")

train_counts = np.bincount(source_train_quintuple)
test_counts = np.bincount(source_test_quintuple)

print(f"\nDataset source breakdown:")
sources = {0: "politic", 1: "climate", 2: "subpop", 3: "nutrition", 4: "policy"}
for src_id, src_name in sources.items():
    if src_id < len(train_counts):
        print(f"  {src_name:10s}: {train_counts[src_id]:>8d} train, {test_counts[src_id]:>8d} test")

print(f"\nTotal:")
print(f"  Train: {len(y_train_quintuple):,} rows")
print(f"  Test:  {len(y_test_quintuple):,} rows")

print(f"\n✅ Ready for quintuple-source probe training!")
print(f"   Files: {OUT_TRAIN_NPZ}, {OUT_TEST_NPZ}")
print(f"\n⚠️  Always condition on dataset_source (0-4) when analyzing!")
print(f"    Each source has different semantics and demographics.")
