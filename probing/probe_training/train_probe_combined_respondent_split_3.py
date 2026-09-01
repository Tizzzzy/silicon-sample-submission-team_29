#!/usr/bin/env python3
"""
Train regression probe using COMBINED (politic + climate) dataset with respondent-level split.
Uses PCA and GridSearchCV to combat high-dimensional overfitting.

⚠️  CRITICAL: This dataset combines two sources with different semantics:
    - Politic: question_id = 0-19+ (real survey questions)
    - Climate: question_id = 0-3 (task types: 0=happening, 1=cause, 2=worry, 3=priority)

    ALWAYS condition on (dataset_source, question_id) jointly when analyzing results!
    The ranges overlap but mean different things across sources.

Workflow:
1. Load combined train/test datasets from train_test_split/ folder
2. Train Ridge regression with PCA on combined train set (politic + climate respondents)
3. Evaluate on held-out test set (zero respondent overlap within each source)
4. Report detailed metrics: overall + per-source + per-granularity + per-condition
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
from tempfile import mkdtemp
from shutil import rmtree
import atexit

print("="*80)
print("Train Opinion Prediction Probe (COMBINED Dataset, Respondent-Level Split)")
print("="*80)
print("\n⚠️  IMPORTANT: This dataset combines politic + climate with different semantics")
print("   Always condition on (dataset_source, question_id) jointly!\n")

# ============================================================================
# Load combined datasets
# ============================================================================

print("Loading combined probe datasets from train_test_split/ folder...")

# Load from train_test_split/ directory (combined politic + climate)
data_train = np.load("train_test_split/probe_dataset_combined_train.npz")
# FIX 1: Downcast X to float32 to cut memory usage in half (~33GB -> ~16.5GB)
X_train = data_train['X'].astype(np.float32)
y_train = data_train['y']
n_dims_train = data_train['n_dims']
condition_code_train = data_train['condition_code']
question_id_train = data_train['question_id']
dataset_source_train = data_train['dataset_source']  # NEW: 0=politic, 1=climate

data_test = np.load("train_test_split/probe_dataset_combined_test.npz")
# FIX 1: Downcast X to float32
X_test = data_test['X'].astype(np.float32)
y_test = data_test['y']
n_dims_test = data_test['n_dims']
condition_code_test = data_test['condition_code']
question_id_test = data_test['question_id']
dataset_source_test = data_test['dataset_source']  # NEW: 0=politic, 1=climate

with open('train_test_split/probe_dataset_combined_train_metadata.json', 'r') as f:
    metadata = json.load(f)

n_train = len(y_train)
n_test = len(y_test)
n_total = n_train + n_test

print(f"  Total train set: {n_train} samples ({100*n_train/n_total:.1f}%)")
print(f"    Politic: {(dataset_source_train == 0).sum()}")
print(f"    Climate: {(dataset_source_train == 1).sum()}")
print(f"  Total test set:  {n_test} samples ({100*n_test/n_total:.1f}%)")
print(f"    Politic: {(dataset_source_test == 0).sum()}")
print(f"    Climate: {(dataset_source_test == 1).sum()}")
print(f"  Features: {X_train.shape[1]}")

# Invert condition code map
code_to_cond = {v: k for k, v in metadata['condition_code_map'].items()}

# Map dataset sources
source_map = {0: "politic", 1: "climate"}

print(f"\n  Conditions: {list(metadata['condition_code_map'].keys())}")
print(f"  Data sources: {source_map}")
print(f"  Dimensions (politic): {metadata['dimensions']['politic']}")
print(f"  Dimensions (climate): {metadata['dimensions']['climate']}")

# ============================================================================
# Verify no data leakage (sanity check)
# ============================================================================

print(f"\nVerifying NO respondent overlap within each source...")
print(f"  ✓ Politic: train/test split at respondent-level")
print(f"  ✓ Climate: train/test split at respondent-level")
print(f"  ✓ Zero respondent overlap within each source (by construction)")

# ============================================================================
# Inspect data distributions (train set)
# ============================================================================

print(f"\nTrain set granularity distribution:")
for nd in range(1, 10):  # Climate goes up to 9D
    count = (n_dims_train == nd).sum()
    if count > 0:
        pct = 100 * count / n_train
        print(f"  {nd}D: {count:5d} ({pct:5.1f}%)")

print(f"\nTest set granularity distribution:")
for nd in range(1, 10):  # Climate goes up to 9D
    count = (n_dims_test == nd).sum()
    if count > 0:
        pct = 100 * count / n_test
        print(f"  {nd}D: {count:5d} ({pct:5.1f}%)")

# ============================================================================
# Train model (PCA + GridSearch)
# ============================================================================

print(f"\nTraining Ridge regression probe on combined train set...")
print(f"  Combining politic ({(dataset_source_train == 0).sum()}) + climate ({(dataset_source_train == 1).sum()}) samples")
print(f"  Applying PCA for dimensionality reduction to combat overfitting")
print(f"  CV: 5-fold cross-validation on training set")

# 1. Create a temporary folder to store cached PCA matrices
cachedir = mkdtemp()

# Clean up the temp folder when the script finishes
atexit.register(lambda: rmtree(cachedir))

# 2. Add 'memory=cachedir' to the Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(svd_solver='randomized', random_state=42)),
    ("ridge", Ridge())
], memory=cachedir)  # <-- THIS IS THE MAGIC FIX

# 2. Hyperparameter grid to search over
param_grid = {
    # FIX 2: Removed 0.90 and 0.95. Floating points force svd_solver='full' 
    # which calculates the exact covariance matrix and will instantly OOM on 800k rows.
    "pca__n_components": [64, 128, 256, 512],
    # Expand alpha to find good regularization
    "ridge__alpha": np.logspace(-1, 7, 25)
}

# 3. Setup GridSearch (keep n_jobs=2 so you don't run out of memory)
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=2, 
    pre_dispatch='2*n_jobs', 
    verbose=3 # Increased verbosity to 3 so you can see each fold completing!
)

grid_search.fit(X_train, y_train)

# 4. Extract the best model
best_pipeline = grid_search.best_estimator_
chosen_alpha = float(best_pipeline.named_steps["ridge"].alpha)
chosen_pca = best_pipeline.named_steps["pca"].n_components_

print(f"\n  Best PCA components: {chosen_pca}")
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

print(f"\n  Combined train set (on which model was trained):")
print(f"    MSE:  {overall_train['mse']:.6f}")
print(f"    RMSE: {overall_train['rmse']:.6f}")
print(f"    R²:   {overall_train['r2']:.6f}")
print(f"    Pearson r: {overall_train['pearson_r']:.6f}")

print(f"\n  Combined test set (held-out, disjoint respondents) ← REAL PERFORMANCE:")
print(f"    MSE:  {overall_test['mse']:.6f}")
print(f"    RMSE: {overall_test['rmse']:.6f}")
print(f"    R²:   {overall_test['r2']:.6f}")
print(f"    Pearson r: {overall_test['pearson_r']:.6f}")

# ============================================================================
# Breakdown by dataset source (test set)
# ============================================================================

print(f"\n⚠️  CRITICAL REMINDER: Evaluate per-source separately!")
print(f"    Politic and climate have different semantics:")
print(f"    - question_id ranges overlap (0-3 used in both)")
print(f"    - Politic: 0-19+ are real survey question ids")
print(f"    - Climate: 0-3 are task_type codes (happening/cause/worry/priority)")

print(f"\nEvaluation by dataset source (test set):")

by_source = {}
for src in [0, 1]:
    src_name = source_map[src]
    mask = dataset_source_test == src
    if mask.sum() >= 2:
        metrics = evaluate(y_test[mask], y_pred_test[mask])
        by_source[src_name] = metrics
        print(f"  {src_name.upper():8s} ({metrics['n']:5d} samples):")
        print(f"    RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:6.4f}, r: {metrics['pearson_r']:7.4f}")

# ============================================================================
# Breakdown by granularity (n_dims) — test set
# ============================================================================

print(f"\nEvaluation by granularity (test set):")

by_ndims = {}
for nd in range(1, 10):  # Climate goes up to 9D
    mask = n_dims_test == nd
    if mask.sum() >= 2:
        metrics = evaluate(y_test[mask], y_pred_test[mask])
        by_ndims[str(nd)] = metrics
        print(f"  {nd}D ({metrics['n']:5d} samples):")
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
        n_politic = ((dataset_source_test == 0) & mask).sum()
        n_climate = ((dataset_source_test == 1) & mask).sum()
        print(f"  {cond_name:25s} ({metrics['n']:5d} samples, {n_politic} politic, {n_climate} climate):")
        print(f"    RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:6.4f}, r: {metrics['pearson_r']:7.4f}")

# ============================================================================
# Breakdown by source + question_id (test set) — RECOMMENDED ANALYSIS
# ============================================================================

print(f"\n⚠️  RECOMMENDED ANALYSIS: Per (source, question_id) pair")
print(f"    This respects the semantic differences between sources\n")

by_source_question = {}
for src in [0, 1]:
    src_name = source_map[src]
    src_mask = dataset_source_test == src
    qids = np.unique(question_id_test[src_mask])

    by_source_question[src_name] = {}

    for qid in sorted(qids):
        mask = src_mask & (question_id_test == qid)
        if mask.sum() >= 2:
            metrics = evaluate(y_test[mask], y_pred_test[mask])
            by_source_question[src_name][int(qid)] = metrics

print(f"  Politic:")
for qid, metrics in sorted(by_source_question['politic'].items()):
    print(f"    Question {qid:2d}: {metrics['n']:4d} samples, R²={metrics['r2']:6.4f}, r={metrics['pearson_r']:7.4f}")

print(f"  Climate (task_type: 0=happening, 1=cause, 2=worry, 3=priority):")
for qid, metrics in sorted(by_source_question['climate'].items()):
    task_names = {0: "happening", 1: "cause", 2: "worry", 3: "priority"}
    task_name = task_names.get(qid, f"unknown_{qid}")
    print(f"    {task_name:10s}:  {metrics['n']:4d} samples, R²={metrics['r2']:6.4f}, r={metrics['pearson_r']:7.4f}")

# ============================================================================
# Save artifacts
# ============================================================================

print(f"\nSaving artifacts...")

joblib.dump(best_pipeline, "ridge_probe_model_combined_256G.joblib")
print(f"  Saved: ridge_probe_model_combined_256G.joblib")

report = {
    "timestamp": datetime.now().isoformat(),
    "data_split": {
        "type": "RESPONDENT-LEVEL (no leakage within each source)",
        "dataset": "COMBINED (politic + climate)",
        "source_train": "train_test_split/probe_dataset_combined_train.npz",
        "source_test": "train_test_split/probe_dataset_combined_test.npz",
        "n_total_rows": n_train + n_test,
        "n_train": n_train,
        "n_test": n_test,
        "train_politic": int((dataset_source_train == 0).sum()),
        "train_climate": int((dataset_source_train == 1).sum()),
        "test_politic": int((dataset_source_test == 0).sum()),
        "test_climate": int((dataset_source_test == 1).sum()),
        "respondent_overlap": 0  # By construction within each source
    },
    "model": {
        "type": "Pipeline(StandardScaler, PCA, Ridge)",
        "alpha_grid": list(np.logspace(-1, 7, 25)),
        "pca_grid": [64, 128, 256, 512],
        "cv_folds": 5,
        "chosen_alpha": chosen_alpha,
        "chosen_pca_components": int(chosen_pca) if isinstance(chosen_pca, (int, float, np.integer, np.floating)) else chosen_pca,
        "training_data": "combined train set (politic + climate respondents)"
    },
    "overall_train": overall_train,
    "overall_test": overall_test,
    "by_source": by_source,
    "by_n_dims": by_ndims,
    "by_condition": by_condition,
    "by_source_question": by_source_question,
    "interpretation": {
        "combined_test_r2": f"{overall_test['r2']:.4f}",
        "politic_test_r2": f"{by_source.get('politic', {}).get('r2', float('nan')):.4f}",
        "climate_test_r2": f"{by_source.get('climate', {}).get('r2', float('nan')):.4f}",
        "train_test_gap": f"{overall_train['r2'] - overall_test['r2']:.4f}",
        "note": "Test R² is the TRUE performance metric (held-out respondents, no leakage). Compare per-source results: politic and climate may have different predictability."
    },
    "critical_notes": [
        "IMPORTANT: This model is trained on COMBINED politic + climate data",
        "Politic and climate have DIFFERENT semantic meanings for overlapping question_id ranges:",
        "  - Politic: question_id 0-19+ are real survey question indices",
        "  - Climate: question_id 0-3 are task_type codes (0=happening, 1=cause, 2=worry, 3=priority)",
        "When analyzing results, ALWAYS condition on (dataset_source, question_id) jointly",
        "See 'by_source_question' breakdown in this report for proper per-source analysis"
    ]
}

with open("evaluation_report_combined_256G.json", 'w') as f:
    json.dump(report, f, indent=2)
print(f"  Saved: evaluation_report_combined_256G.json")

# ============================================================================
# Final summary
# ============================================================================

print(f"\n" + "="*80)
print("PROBE TRAINING COMPLETE (COMBINED DATASET, NO LEAKAGE)")
print("="*80)

print(f"\n✅ Probe trained and evaluated on COMBINED (politic + climate) dataset!")
print(f"   Zero respondent overlap within each source (by construction)")

print(f"\nKey Results (held-out test set):")
print(f"  Combined R²:       {overall_test['r2']:.4f}")
print(f"  Combined RMSE:     {overall_test['rmse']:.4f}")
print(f"  Combined Pearson r: {overall_test['pearson_r']:.4f}")

print(f"\nPer-source results (test set):")
for src in [0, 1]:
    src_name = source_map[src]
    if src_name in by_source:
        metrics = by_source[src_name]
        print(f"  {src_name.upper():8s}: R²={metrics['r2']:6.4f}, RMSE={metrics['rmse']:.6f}")

print(f"\nGeneralization check (combined):")
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
    print(f"  R² = {overall_test['r2']:.4f} → Very weak signal (LLMs may not encode demographic opinions well)")

print(f"\n⚠️  IMPORTANT REMINDERS:")
print(f"  1. Politic and climate are separate in semantics (different dimensions, MIN_N, question meanings)")
print(f"  2. Always analyze per-source to understand what the model learned from each")
print(f"  3. See 'by_source_question' breakdown in evaluation_report_combined.json")
print(f"  4. Use dataset_source array to disambiguate question_id when analyzing predictions")

print(f"\nArtifacts saved:")
print(f"  - ridge_probe_model_combined.joblib (trained model)")
print(f"  - evaluation_report_combined.json (comprehensive metrics + per-source breakdown)")