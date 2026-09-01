#!/usr/bin/env python3
"""
Train regression probe to predict group opinion means from averaged representations.

Workflow:
1. Load pre-built probe dataset (NPZ + metadata)
2. Perform stratified train/test split (80/20, stratified by n_dims for balanced coverage)
3. Train Ridge regression with cross-validated alpha selection
4. Evaluate on held-out test set
5. Report detailed metrics: overall + per-granularity + per-condition
"""

import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib
from datetime import datetime

print("="*80)
print("Train Opinion Prediction Probe")
print("="*80)

# ============================================================================
# Load dataset
# ============================================================================

print("\nLoading probe dataset...")

data = np.load("/projects/p32143/silicon-sample-submission/probing/probe_training/probe_dataset.npz")
X = data['X']  # (N, 5120)
y = data['y']  # (N,)
n_dims = data['n_dims']  # (N,)
condition_code = data['condition_code']  # (N,)
question_id = data['question_id']  # (N,)

with open('/projects/p32143/silicon-sample-submission/probing/probe_training/probe_dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

N = len(y)
print(f"  Loaded {N} samples")
print(f"  Features: {X.shape[1]}")

# Invert condition code map
code_to_cond = {v: k for k, v in metadata['condition_code_map'].items()}

# ============================================================================
# Stratified train/test split
# ============================================================================

print("\nPerforming stratified train/test split (80/20, stratified by n_dims)...")

idx = np.arange(N)
idx_train, idx_test = train_test_split(
    idx,
    test_size=0.2,
    random_state=42,
    stratify=n_dims
)

X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]
n_dims_train, n_dims_test = n_dims[idx_train], n_dims[idx_test]
condition_train, condition_test = condition_code[idx_train], condition_code[idx_test]
question_train, question_test = question_id[idx_train], question_id[idx_test]

print(f"  Train set: {len(idx_train)} samples ({100*len(idx_train)/N:.1f}%)")
print(f"  Test set:  {len(idx_test)} samples ({100*len(idx_test)/N:.1f}%)")

print(f"\n  Train set granularity distribution:")
for nd in range(1, 5):
    count = (n_dims_train == nd).sum()
    pct = 100 * count / len(idx_train)
    print(f"    {nd}D: {count:5d} ({pct:5.1f}%)")

print(f"\n  Test set granularity distribution:")
for nd in range(1, 5):
    count = (n_dims_test == nd).sum()
    pct = 100 * count / len(idx_test)
    print(f"    {nd}D: {count:5d} ({pct:5.1f}%)")

# ============================================================================
# Train model
# ============================================================================

print(f"\nTraining Ridge regression probe...")
print(f"  Alphas: 10^-2 to 10^6 (33 values)")
print(f"  CV: 5-fold cross-validation on training set")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(
        alphas=np.logspace(-2, 6, 33),
        cv=5,
        scoring='r2'
    ))
])

pipeline.fit(X_train, y_train)

chosen_alpha = float(pipeline.named_steps["ridge"].alpha_)
print(f"  Chosen alpha: {chosen_alpha:.6f}")

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

print(f"\nEvaluating on test set...")

y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

overall_train = evaluate(y_train, y_pred_train)
overall_test = evaluate(y_test, y_pred_test)

print(f"\n  Train set:")
print(f"    MSE:  {overall_train['mse']:.6f}")
print(f"    RMSE: {overall_train['rmse']:.6f}")
print(f"    R²:   {overall_train['r2']:.6f}")
print(f"    Pearson r: {overall_train['pearson_r']:.6f}")

print(f"\n  Test set:")
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
for code in sorted(np.unique(condition_test)):
    mask = condition_test == code
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

# Save model
joblib.dump(pipeline, "ridge_probe_model.joblib")
print(f"  Saved: ridge_probe_model.joblib")

# Save evaluation report
report = {
    "timestamp": datetime.now().isoformat(),
    "build_info": {
        "source_npz": "probe_dataset.npz",
        "source_metadata": "probe_dataset_metadata.json",
        "n_total_rows": int(N),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "test_size": 0.2,
        "random_state": 42,
        "stratify_by": "n_dims"
    },
    "model": {
        "type": "Pipeline(StandardScaler, RidgeCV)",
        "alpha_grid": list(np.logspace(-2, 6, 33)),
        "cv_folds": 5,
        "chosen_alpha": chosen_alpha
    },
    "overall_train": overall_train,
    "overall_test": overall_test,
    "by_n_dims": by_ndims,
    "by_condition": by_condition,
    "notes": "Evaluated on held-out test set; condition is part of the group key; ground_truth y was recomputed fresh per group from individual_opinion values."
}

with open("evaluation_report.json", 'w') as f:
    json.dump(report, f, indent=2)
print(f"  Saved: evaluation_report.json")

# ============================================================================
# Final summary
# ============================================================================

print(f"\n" + "="*80)
print("PROBE TRAINING COMPLETE")
print("="*80)

print(f"\n✅ Probe successfully trained and evaluated!")
print(f"\nKey Results:")
print(f"  Test set R²:       {overall_test['r2']:.4f}")
print(f"  Test set RMSE:     {overall_test['rmse']:.4f}")
print(f"  Test set Pearson r: {overall_test['pearson_r']:.4f}")
print(f"\nArtifacts saved:")
print(f"  - ridge_probe_model.joblib (trained model + scaler)")
print(f"  - evaluation_report.json (detailed metrics by granularity & condition)")
print(f"\nReadiness Check:")
print(f"  Can the probe predict group opinions from representations?")
if overall_test['r2'] > 0.3:
    print(f"  ✓ YES — R² = {overall_test['r2']:.4f} suggests moderate predictive signal")
elif overall_test['r2'] > 0.1:
    print(f"  ≈ WEAK — R² = {overall_test['r2']:.4f} suggests weak predictive signal")
else:
    print(f"  ✗ NO — R² = {overall_test['r2']:.4f} suggests very weak signal")
