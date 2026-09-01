#!/usr/bin/env python3
"""
Inference script for quintuple-source probe.

Loads trained PCA + Ridge models and makes predictions on new data.
Can be used for:
- Evaluating on arbitrary representation data
- Analyzing specific demographic groups
- Computing predictions on validation sets
"""

import pickle
import numpy as np
import sys
import os

print("="*80)
print("QUINTUPLE-SOURCE PROBE INFERENCE")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

MODEL_PCA_FILE = "probe_model_pca_quintuple.pkl"
MODEL_RIDGE_FILE = "probe_model_ridge_quintuple.pkl"

# Validate that model files exist
if not os.path.exists(MODEL_PCA_FILE):
    print(f"\n❌ Error: PCA model not found at {MODEL_PCA_FILE}")
    print(f"   Please run train_probe_quintuple_respondent_split.py first")
    sys.exit(1)

if not os.path.exists(MODEL_RIDGE_FILE):
    print(f"\n❌ Error: Ridge model not found at {MODEL_RIDGE_FILE}")
    print(f"   Please run train_probe_quintuple_respondent_split.py first")
    sys.exit(1)

# ============================================================================
# Load Trained Models
# ============================================================================

print(f"\nLoading trained models...")

with open(MODEL_PCA_FILE, 'rb') as f:
    pca = pickle.load(f)

with open(MODEL_RIDGE_FILE, 'rb') as f:
    ridge = pickle.load(f)

print(f"✓ PCA model loaded")
print(f"  Input dimension: 5120")
print(f"  Output dimension: {pca.n_components_}")

print(f"✓ Ridge model loaded")
print(f"  Alpha: {ridge.alpha:.4f}")

# ============================================================================
# Inference Function
# ============================================================================

def predict_opinions(X):
    """
    Predict opinions from representation vectors.

    Args:
        X: numpy array of shape (N, 5120) with representation vectors

    Returns:
        y_pred: numpy array of shape (N,) with predicted opinions
    """
    if X.shape[1] != 5120:
        raise ValueError(f"Expected 5120-dim vectors, got {X.shape[1]}")

    X_pca = pca.transform(X.astype(np.float32))
    y_pred = ridge.predict(X_pca)

    return y_pred


# ============================================================================
# Example Usage
# ============================================================================

print(f"\n{'='*80}")
print("Example Usage")
print(f"{'='*80}")

print(f"""
# Load representations (e.g., from a test set)
import numpy as np
X_test = np.load('some_representations.npz')['X']  # shape: (N, 5120)

# Make predictions
y_pred = predict_opinions(X_test)

# Evaluate (if ground truth is available)
from sklearn.metrics import r2_score
y_true = np.load('some_representations.npz')['y']
r2 = r2_score(y_true, y_pred)
print(f'R²: {{r2:.4f}}')

# Analyze specific groups
source = np.load('some_representations.npz')['dataset_source']
for src_id in range(5):
    mask = source == src_id
    r2_src = r2_score(y_true[mask], y_pred[mask])
    print(f'Source {{src_id}}: R² = {{r2_src:.4f}}')
""")

print(f"\n{'='*80}")
print("Functions Available")
print(f"{'='*80}")

print(f"""
predict_opinions(X)
  Make predictions on representation vectors

  Args:
    X: (N, 5120) array of representation vectors

  Returns:
    y_pred: (N,) array of predicted opinions

  Example:
    y_pred = predict_opinions(X_test)
""")

print(f"\nTo use in your own script:")
print(f"""
# Copy this into your script and run:
import pickle
import numpy as np

# Load models
with open('{MODEL_PCA_FILE}', 'rb') as f:
    pca = pickle.load(f)
with open('{MODEL_RIDGE_FILE}', 'rb') as f:
    ridge = pickle.load(f)

# Define inference function
def predict_opinions(X):
    X_pca = pca.transform(X.astype(np.float32))
    return ridge.predict(X_pca)

# Use it
X_test = np.load('your_data.npz')['X']
y_pred = predict_opinions(X_test)
""")

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("Ready for Inference")
    print(f"{'='*80}")
    print(f"\nImport this script or use it as a template for your inference code")
