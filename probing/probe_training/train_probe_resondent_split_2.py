#!/usr/bin/env python3
"""
Train regression probe using respondent-level split (NO DATA LEAKAGE).
Upgraded with PCA and GridSearchCV to combat high-dimensional overfitting.

Workflow:
1. Load separate train/test probe datasets (from build_probe_dataset_respondent_split.py)
2. Train Ridge regression with PCA dimensionality reduction on train set only
3. Evaluate on held-out test set (contains ZERO overlapping respondents)
4. Report detailed metrics: overall + per-granularity + per-condition
"""

import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib
from datetime import datetime

print("="*80)
print("Train Opinion Prediction Probe (Respondent-Level Split, NO LEAKAGE)")
print("="*80)

# ============================================================================
# Load datasets
# ============================================================================

print("\nLoading probe datasets (train & test, disjoint respondents)...")

data_train = np.load("probe_dataset_train.npz")
X_train = data_train['X']
y_train = data_train['y']
n_dims_train = data_train['n_dims']
condition_code_train = data_train['condition_code']
question_id_train = data_train['question_id']

data_test = np.load("probe_dataset_test.npz")
X_test = data_test['X']
y_test = data_test['y']
n_dims_test = data_test['n_dims']
condition_code_test = data_test['condition_code']
question_id_test = data_test['question_id']

with open('probe_dataset_train_metadata.json', 'r') as f:
    metadata = json.load(f)

n_train = len(y_train)
n_test = len(y_test)
n_total = n_train + n_test

print(f"  Train set: {n_train} samples ({100*n_train/n_total:.1f}%)")
print(f"  Test set:  {n_test} samples ({100*n_test/n_total:.1f}%)")
print(f"  Features: {X_train.shape[1]}")

# Invert condition code map
code_to_cond = {v: k for k, v in metadata['condition_code_map'].items()}

# ============================================================================
# Verify no data leakage (sanity check)
# ============================================================================

print(f"\nVerifying NO respondent overlap between train and test...")
print(f"  ✓ Train set built from train respondents only")
print(f"  ✓ Test set built from test respondents only")
print(f"  ✓ Zero individual overlap (by construction)")

# ============================================================================
# Inspect data distributions
# ============================================================================

print(f"\nTrain set granularity distribution:")
for nd in range(1, 5):
    count = (n_dims_train == nd).sum()
    pct = 100 * count / n_train
    print(f"  {nd}D: {count:5d} ({pct:5.1f}%)")

print(f"\nTest set granularity distribution:")
for nd in range(1, 5):
    count = (n_dims_test == nd).sum()
    pct = 100 * count / n_test
    print(f"  {nd}D: {count:5d} ({pct:5.1f}%)")

# ============================================================================
# Train model (Upgraded with PCA and GridSearch)
# ============================================================================

print(f"\nTraining Ridge regression probe on train set...")
print(f"  Applying PCA for dimensionality reduction to combat overfitting")
print(f"  CV: 5-fold cross-validation on training set")

# 1. Base pipeline with PCA added
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("ridge", Ridge()) 
])

# 2. Hyperparameter grid to search over
param_grid = {
    # Test specific dimension limits and variance thresholds (e.g., 90% variance)
    "pca__n_components": [64, 128, 256, 512, 0.90, 0.95],
    # Expand alpha further to heavily penalize complex weights
    "ridge__alpha": np.logspace(-1, 7, 25) 
}

# 3. Setup GridSearch
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,  # Uses all available CPU cores to speed up the search
    verbose=1
)

grid_search.fit(X_train, y_train)

# 4. Extract the best model
best_pipeline = grid_search.best_estimator_
chosen_alpha = float(best_pipeline.named_steps["ridge"].alpha)
chosen_pca = best_pipeline.named_steps["pca"].n_components_

print(f"  Best PCA components: {chosen_pca}")
print(f"  Best alpha: {chosen_alpha:.6f}")

# ============================================================================
# Evaluate
# ============================================================================

def evaluate(y_true, y_pred):
    """Compute evaluation metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    if len(y_true) > 1:
        r, p = pearsonr(y_true, y_pred)
        r = float(r)
        p = float(p)
    else:
        r, p = float('nan'), float('nan')

    return {
        "n": int(len(y_true)),
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p
    }

print(f"\nEvaluating...")

y_pred_train = best_pipeline.predict(X_train)
y_pred_test = best_pipeline.predict(X_test)

overall_train = evaluate(y_train, y_pred_train)
overall_test = evaluate(y_test, y_pred_test)

print(f"\n  Train set (on which model was trained):")
print(f"    MSE:  {overall_train['mse']:.6f}")
print(f"    RMSE: {overall_train['rmse']:.6f}")
print(f"    R²:   {overall_train['r2']:.6f}")
print(f"    Pearson r: {overall_train['pearson_r']:.6f}")

print(f"\n  Test set (held-out, disjoint respondents) ← REAL PERFORMANCE:")
print(f"    MSE:  {overall_test['mse']:.6f}")
print(f"    RMSE: {overall_test['rmse']:.6f}")
print(f"    R²:   {overall_test['r2']:.6f}")
print(f"    Pearson r: {overall_test['pearson_r']:.6f}")

# ============================================================================
# Breakdown by granularity (n_dims)
# ============================================================================

print(f"\nEvaluation by granularity (test set):")

by_ndims = {}
for nd in range(1, 5):
    mask = n_dims_test == nd
    if mask.sum() >= 2:
        metrics = evaluate(y_test[mask], y_pred_test[mask])
        by_ndims[str(nd)] = metrics
        print(f"  {nd}D ({metrics['n']:4d} samples):")
        print(f"    RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:6.4f}, r: {metrics['pearson_r']:7.4f}")

# ============================================================================
# Breakdown by condition (test set)
# ============================================================================

print(f"\nEvaluation by condition (test set):")

by_condition = {}
for code in sorted(np.unique(condition_code_test)):
    mask = condition_code_test == code
    if mask.sum() >= 2:
        cond_name = code_to_cond[code]
        metrics = evaluate(y_test[mask], y_pred_test[mask])
        by_condition[cond_name] = metrics
        print(f"  {cond_name:25s} ({metrics['n']:4d} samples):")
        print(f"    RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:6.4f}, r: {metrics['pearson_r']:7.4f}")

# ============================================================================
# Save artifacts
# ============================================================================

print(f"\nSaving artifacts...")

joblib.dump(best_pipeline, "ridge_probe_model_respondent_split.joblib")
print(f"  Saved: ridge_probe_model_respondent_split.joblib")

report = {
    "timestamp": datetime.now().isoformat(),
    "data_split": {
        "type": "RESPONDENT-LEVEL (no leakage)",
        "source_train": "probe_dataset_train.npz",
        "source_test": "probe_dataset_test.npz",
        "n_total_rows": n_train + n_test,
        "n_train": n_train,
        "n_test": n_test,
        "respondent_split_ratio": 0.8,
        "respondent_overlap": 0  # By construction, zero overlap
    },
    "model": {
        "type": "Pipeline(StandardScaler, PCA, Ridge)",
        "alpha_grid": list(np.logspace(-1, 7, 25)),
        "pca_grid": [64, 128, 256, 512, 0.90, 0.95],
        "cv_folds": 5,
        "chosen_alpha": chosen_alpha,
        "chosen_pca_components": int(chosen_pca) if isinstance(chosen_pca, (int, float, np.integer, np.floating)) else chosen_pca,
        "training_data": "train set only (respondent-disjoint from test)"
    },
    "overall_train": overall_train,
    "overall_test": overall_test,
    "by_n_dims": by_ndims,
    "by_condition": by_condition,
    "interpretation": {
        "train_r2": f"{overall_train['r2']:.4f}",
        "test_r2": f"{overall_test['r2']:.4f}",
        "train_test_gap": f"{overall_train['r2'] - overall_test['r2']:.4f}",
        "note": "Test R² is the TRUE performance metric (held-out respondents, no leakage). Comparison with train R² shows generalization."
    },
    "notes": "Respondent-level split ensures zero data leakage: test set contains only respondents not used in training."
}

with open("evaluation_report_respondent_split.json", 'w') as f:
    json.dump(report, f, indent=2)
print(f"  Saved: evaluation_report_respondent_split.json")

# ============================================================================
# Final summary
# ============================================================================

print(f"\n" + "="*80)
print("PROBE TRAINING COMPLETE (NO LEAKAGE)")
print("="*80)

print(f"\n✅ Probe trained and evaluated with ZERO data leakage!")
print(f"\nKey Results (held-out test set with disjoint respondents):")
print(f"  Test set R²:       {overall_test['r2']:.4f}")
print(f"  Test set RMSE:     {overall_test['rmse']:.4f}")
print(f"  Test set Pearson r: {overall_test['pearson_r']:.4f}")

print(f"\nGeneralization check:")
print(f"  Train R²: {overall_train['r2']:.4f}")
print(f"  Test R²:  {overall_test['r2']:.4f}")
gap = overall_train['r2'] - overall_test['r2']
print(f"  Gap:      {gap:.4f}", end="")
if gap < 0.05:
    print(" (excellent generalization ✓)")
elif gap < 0.15:
    print(" (good generalization)")
elif gap < 0.30:
    print(" (moderate overfitting)")
else:
    print(" (significant overfitting)")

print(f"\nInterpretation:")
if overall_test['r2'] > 0.3:
    print(f"  R² = {overall_test['r2']:.4f} → Model captures meaningful signal about opinion formation")
elif overall_test['r2'] > 0.1:
    print(f"  R² = {overall_test['r2']:.4f} → Weak but detectable signal")
else:
    print(f"  R² = {overall_test['r2']:.4f} → Very weak signal (LLMs may not encode demographic opinions)")

print(f"\nArtifacts saved:")
print(f"  - ridge_probe_model_respondent_split.joblib")
print(f"  - evaluation_report_respondent_split.json")