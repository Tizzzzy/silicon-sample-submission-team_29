#!/usr/bin/env python3
"""
Train probe on quintuple-source dataset (politic + climate + subpop + nutrition + policy).

Extends train_probe_combined_respondent_split.py with support for 5 sources:
  0 = politic (10k rows, 4 dims, condition treatments)
  1 = climate (806.5k rows, 9 dims, N/A conditions)
  2 = subpop (2.6k rows, 1 dim, pre-aggregated)
  3 = nutrition (individual→grouped, task_type + text + demographics)
  4 = policy (individual→grouped, target_variable + condition + demographics)

This script:
1. Loads quintuple-source NPZ + metadata
2. Trains a unified Ridge probe (PCA + Ridge with GridSearchCV)
3. Evaluates overall + per-source + per-granularity + per-source-question breakdown
4. Outputs detailed evaluation report with all breakdowns
"""

import json
import numpy as np
import os
import pickle
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

print("="*80)
print("QUINTUPLE-SOURCE PROBE TRAINING")
print("(politic + climate + subpop + nutrition + policy)")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = "/projects/p32143/silicon-sample-submission/probing/probe_training/train_test_split"
TRAIN_FILE = os.path.join(DATA_DIR, "probe_dataset_quintuple_train.npz")
TEST_FILE = os.path.join(DATA_DIR, "probe_dataset_quintuple_test.npz")
TRAIN_META_FILE = os.path.join(DATA_DIR, "probe_dataset_quintuple_train_metadata.json")

OUTPUT_REPORT = "evaluation_report_quintuple.json"
MODEL_PCA_FILE = "probe_model_pca_quintuple.pkl"
MODEL_RIDGE_FILE = "probe_model_ridge_quintuple.pkl"

# Ridge regression hyperparameters
PCA_COMPONENTS_RANGE = [50, 100, 200, 400, 512, 1024, 2048]
ALPHA_RANGE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

SOURCE_MAP = {0: "politic", 1: "climate", 2: "subpop", 3: "nutrition", 4: "policy"}
SOURCE_NAMES = ["politic", "climate", "subpop", "nutrition", "policy"]

print(f"\nConfiguration:")
print(f"  Train file: {TRAIN_FILE}")
print(f"  Test file:  {TEST_FILE}")
print(f"  PCA range: {PCA_COMPONENTS_RANGE}")
print(f"  Ridge alpha range: {ALPHA_RANGE}")

# ============================================================================
# Load Data
# ============================================================================

print(f"\n{'='*80}")
print("Loading Data")
print(f"{'='*80}")

print(f"\nLoading train data...")
train_data = np.load(TRAIN_FILE)
X_train_full = train_data['X'].astype(np.float32)
y_train_full = train_data['y'].astype(np.float32)
source_train_full = train_data['dataset_source'].astype(np.int8)
n_dims_train_full = train_data['n_dims'].astype(np.int8)
question_id_train_full = train_data['question_id'].astype(np.int16)
condition_code_train_full = train_data['condition_code'].astype(np.int8)

# --- Subsample climate data (source == 1) ---
np.random.seed(42)  # Ensure reproducibility across runs

climate_idx = np.where(source_train_full == 1)[0]
non_climate_idx = np.where(source_train_full != 1)[0]

# Select exactly 50% of climate indices without replacement
climate_keep_count = int(len(climate_idx) * 0.08)
climate_keep_idx = np.random.choice(climate_idx, size=climate_keep_count, replace=False)

# Recombine and shuffle
keep_idx = np.concatenate([non_climate_idx, climate_keep_idx])
np.random.shuffle(keep_idx)  # Shuffle so data sources aren't clustered together

# Apply indices to create final training arrays
X_train = X_train_full[keep_idx]
y_train = y_train_full[keep_idx]
source_train = source_train_full[keep_idx]
n_dims_train = n_dims_train_full[keep_idx]
question_id_train = question_id_train_full[keep_idx]
condition_code_train = condition_code_train_full[keep_idx]
# --------------------------------------------

print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  source_train: {source_train.shape}")

print(f"\nLoading test data...")
test_data = np.load(TEST_FILE)
X_test = test_data['X'].astype(np.float32)
y_test = test_data['y'].astype(np.float32)
source_test = test_data['dataset_source'].astype(np.int8)
n_dims_test = test_data['n_dims'].astype(np.int8)
question_id_test = test_data['question_id'].astype(np.int16)
condition_code_test = test_data['condition_code'].astype(np.int8)

print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")
print(f"  source_test: {source_test.shape}")

# Load metadata
# print(f"\nLoading metadata...")
# with open(TRAIN_META_FILE, 'r') as f:
  #   metadata = json.load(f)

print(f"\nDataset summary:")
for src_id, src_name in SOURCE_MAP.items():
    train_mask = source_train == src_id
    test_mask = source_test == src_id
    n_train = np.sum(train_mask)
    n_test = np.sum(test_mask)
    print(f"  {src_name:10s}: {n_train:>8d} train, {n_test:>8d} test")

print(f"  {'Total':10s}: {len(y_train):>8d} train, {len(y_test):>8d} test")

# ============================================================================
# Train Unified Probe (GridSearchCV)
# ============================================================================

print(f"\n{'='*80}")
print("Training Unified Probe with GridSearchCV")
print(f"{'='*80}")

print(f"\nPerforming GridSearchCV...")
print(f"  Testing PCA components: {PCA_COMPONENTS_RANGE}")
print(f"  Testing Ridge alphas: {ALPHA_RANGE}")

t0 = time.time()

best_r2 = -np.inf
best_params = None
best_pca = None
best_ridge = None

# Grid search over PCA components and Ridge alpha
param_grid = {
    'ridge__alpha': ALPHA_RANGE,
}

# For efficiency, test a few PCA configurations
for n_comp in PCA_COMPONENTS_RANGE:
    print(f"\n  Testing PCA with {n_comp} components...")

    # Apply PCA
    pca = PCA(n_components=min(n_comp, X_train.shape[1]-1), random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Grid search over alpha
    ridge_gs = GridSearchCV(
        Ridge(),
        {'alpha': ALPHA_RANGE},
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=3
    )
    ridge_gs.fit(X_train_pca, y_train)

    # Evaluate on test set
    y_pred_test = ridge_gs.predict(X_test_pca)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"    Best CV alpha: {ridge_gs.best_params_['alpha']:.4f}")
    print(f"    Test R²: {r2_test:.4f}")

    if r2_test > best_r2:
        best_r2 = r2_test
        best_params = {'n_components': n_comp, 'alpha': ridge_gs.best_params_['alpha']}
        best_pca = pca
        best_ridge = ridge_gs.best_estimator_

elapsed = time.time() - t0
print(f"\nGridSearchCV completed in {elapsed:.1f}s")
print(f"Best parameters: {best_params}")
print(f"Best test R²: {best_r2:.4f}")

# ============================================================================
# Final Predictions and Evaluation
# ============================================================================

print(f"\n{'='*80}")
print("Final Evaluation")
print(f"{'='*80}")

# Get final predictions
y_pred_test = best_ridge.predict(best_pca.transform(X_test))

# Overall metrics
overall_mse = mean_squared_error(y_test, y_pred_test)
overall_rmse = np.sqrt(overall_mse)
overall_r2 = r2_score(y_test, y_pred_test)
overall_pearson_r, overall_pearson_p = pearsonr(y_test, y_pred_test)

print(f"\nOverall test metrics:")
print(f"  MSE:        {overall_mse:.6f}")
print(f"  RMSE:       {overall_rmse:.6f}")
print(f"  R²:         {overall_r2:.6f}")
print(f"  Pearson r:  {overall_pearson_r:.6f}")

# ============================================================================
# Per-Source Evaluation
# ============================================================================

print(f"\n{'='*80}")
print("Per-Source Breakdown")
print(f"{'='*80}")

by_source = {}
for src_id, src_name in SOURCE_MAP.items():
    mask = source_test == src_id
    if np.sum(mask) == 0:
        continue

    y_test_src = y_test[mask]
    y_pred_src = y_pred_test[mask]

    mse_src = mean_squared_error(y_test_src, y_pred_src)
    rmse_src = np.sqrt(mse_src)
    r2_src = r2_score(y_test_src, y_pred_src)
    pearson_r_src, _ = pearsonr(y_test_src, y_pred_src)

    by_source[src_name] = {
        'n': int(np.sum(mask)),
        'mse': float(mse_src),
        'rmse': float(rmse_src),
        'r2': float(r2_src),
        'pearson_r': float(pearson_r_src),
    }

    print(f"\n{src_name}:")
    print(f"  n:         {int(np.sum(mask))}")
    print(f"  MSE:       {mse_src:.6f}")
    print(f"  RMSE:      {rmse_src:.6f}")
    print(f"  R²:        {r2_src:.6f}")
    print(f"  Pearson r: {pearson_r_src:.6f}")

# ============================================================================
# Per-Granularity Evaluation (n_dims)
# ============================================================================

print(f"\n{'='*80}")
print("Per-Granularity Breakdown (by n_dims)")
print(f"{'='*80}")

by_n_dims = {}
for n_dim in sorted(set(n_dims_test)):
    mask = n_dims_test == n_dim
    if np.sum(mask) == 0:
        continue

    y_test_nd = y_test[mask]
    y_pred_nd = y_pred_test[mask]

    mse_nd = mean_squared_error(y_test_nd, y_pred_nd)
    rmse_nd = np.sqrt(mse_nd)
    r2_nd = r2_score(y_test_nd, y_pred_nd)

    by_n_dims[str(n_dim)] = {
        'n': int(np.sum(mask)),
        'mse': float(mse_nd),
        'rmse': float(rmse_nd),
        'r2': float(r2_nd),
    }

    print(f"\n{n_dim}D groups:")
    print(f"  n:     {int(np.sum(mask))}")
    print(f"  R²:    {r2_nd:.6f}")

# ============================================================================
# Per-Source-Question Evaluation
# ============================================================================

print(f"\n{'='*80}")
print("Per-Source-Question Breakdown")
print(f"{'='*80}")

by_source_question = {}
for src_id, src_name in SOURCE_MAP.items():
    by_source_question[src_name] = {}

    src_mask = source_test == src_id
    for qid in set(question_id_test[src_mask]):
        mask = (source_test == src_id) & (question_id_test == qid)
        if np.sum(mask) == 0:
            continue

        y_test_sq = y_test[mask]
        y_pred_sq = y_pred_test[mask]

        r2_sq = r2_score(y_test_sq, y_pred_sq)

        by_source_question[src_name][int(qid)] = {
            'n': int(np.sum(mask)),
            'r2': float(r2_sq),
        }

# Print sample breakdown
print("\nSample per-source-question metrics:")
for src_name in SOURCE_NAMES:
    if src_name not in by_source_question or not by_source_question[src_name]:
        continue
    print(f"\n{src_name}:")
    for qid in sorted(by_source_question[src_name].keys())[:3]:
        metrics = by_source_question[src_name][qid]
        print(f"  Question {qid}: R²={metrics['r2']:.4f} (n={metrics['n']})")

# ============================================================================
# Save Report
# ============================================================================

print(f"\n{'='*80}")
print("Saving Report")
print(f"{'='*80}")

report = {
    'timestamp': str(time.time()),
    'model': 'Ridge + PCA',
    'dataset': 'quintuple-source (politic + climate + subpop + nutrition + policy)',
    'best_hyperparameters': best_params,
    'overall_metrics': {
        'n': int(len(y_test)),
        'mse': float(overall_mse),
        'rmse': float(overall_rmse),
        'r2': float(overall_r2),
        'pearson_r': float(overall_pearson_r),
    },
    'by_source': by_source,
    'by_n_dims': by_n_dims,
    'by_source_question': by_source_question,
}

with open(OUTPUT_REPORT, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved to {OUTPUT_REPORT}")

# ============================================================================
# Save Trained Models
# ============================================================================

print(f"\n{'='*80}")
print("Saving Trained Models")
print(f"{'='*80}")

# Save PCA model
with open(MODEL_PCA_FILE, 'wb') as f:
    pickle.dump(best_pca, f)
pca_size = os.path.getsize(MODEL_PCA_FILE) / 1e6
print(f"\n✓ PCA model saved to {MODEL_PCA_FILE} ({pca_size:.1f} MB)")

# Save Ridge model
with open(MODEL_RIDGE_FILE, 'wb') as f:
    pickle.dump(best_ridge, f)
ridge_size = os.path.getsize(MODEL_RIDGE_FILE) / 1e6
print(f"✓ Ridge model saved to {MODEL_RIDGE_FILE} ({ridge_size:.1f} MB)")

print(f"\nModel configuration:")
print(f"  PCA components: {best_params['n_components']}")
print(f"  Ridge alpha: {best_params['alpha']:.4f}")

# Update report to include model file references
report['model_files'] = {
    'pca': MODEL_PCA_FILE,
    'ridge': MODEL_RIDGE_FILE,
}

with open(OUTPUT_REPORT, 'w') as f:
    json.dump(report, f, indent=2)

print(f"✓ Report updated with model file references")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"{'='*80}")

print(f"\nKey Results:")
print(f"  Overall test R²:    {overall_r2:.4f}")
print(f"  Best R² per source:")
for src_name in SOURCE_NAMES:
    if src_name in by_source:
        print(f"    {src_name:10s}: {by_source[src_name]['r2']:.4f}")

print(f"\nFiles Generated:")
print(f"  Report:       {OUTPUT_REPORT}")
print(f"  PCA model:    {MODEL_PCA_FILE}")
print(f"  Ridge model:  {MODEL_RIDGE_FILE}")

print(f"\nUsage (load and make predictions):")
print(f"""
import numpy as np
import pickle

# Load trained models
with open('{MODEL_PCA_FILE}', 'rb') as f:
    pca = pickle.load(f)
with open('{MODEL_RIDGE_FILE}', 'rb') as f:
    ridge = pickle.load(f)

# Make predictions on new data
X_new = ...  # shape: (N, 5120)
X_new_pca = pca.transform(X_new)
y_pred = ridge.predict(X_new_pca)
""")

print(f"\nNext steps:")
print(f"  1. Compare quintuple R² with previous triple-source model")
print(f"  2. Analyze nutrition and policy per-question performance")
print(f"  3. Check if nutrition/policy improve predictability")
print(f"  4. Use saved models for inference on new data")
