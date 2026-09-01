#!/usr/bin/env python3
"""
Generate predictions for cell group representations using trained probe models.

Input:
  - group_representations_main.json (221 cells × 5120-dim vectors)
  - group_representations_moderator.json (5,967 cells × 5120-dim vectors)
  - Trained PCA model (probe_model_pca_quintuple.pkl)
  - Trained Ridge model (probe_model_ridge_quintuple.pkl)

Output:
  - T2_primary_v1_cells_main.csv (condition, outcome, mean)
  - T2_primary_v1_cells_moderator.csv (condition, moderator, level, outcome, mean)
  - prediction_report_quintuple.json (detailed metrics and per-source breakdown)

Usage:
  python predict_cell_groups_quintuple.py \\
    --representations_main ../extract_representation/group_representations_main.json \\
    --representations_moderator ../extract_representation/group_representations_moderator.json \\
    --pca_model ../probe_training/probe_model_pca_quintuple.pkl \\
    --ridge_model ../probe_training/probe_model_ridge_quintuple.pkl \\
    --output_dir .
"""

import argparse
import json
import logging
import sys
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

log = logging.getLogger("predict_cell_groups")


def setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(console)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations_main", default="../extract_representation/group_representations_main.json",
                       help="Path to main cell group representations JSON")
    parser.add_argument("--representations_moderator", default="../extract_representation/group_representations_moderator.json",
                       help="Path to moderator cell group representations JSON")
    parser.add_argument("--pca_model", default="/projects/p32143/silicon-sample-submission/probing/probe_training/probe_model_pca_quintuple.pkl",
                       help="Path to trained PCA model")
    parser.add_argument("--ridge_model", default="/projects/p32143/silicon-sample-submission/probing/probe_training/probe_model_ridge_quintuple.pkl",
                       help="Path to trained Ridge model")
    parser.add_argument("--output_dir", default=".",
                       help="Output directory for prediction CSV and report files")
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 78)
    log.info("PREDICT CELL GROUP REPRESENTATIONS (QUINTUPLE PROBE)")
    log.info("=" * 78)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained models
    log.info(f"\nLoading trained models...")
    with open(args.pca_model, 'rb') as f:
        pca = pickle.load(f)
    log.info(f"  ✓ PCA model loaded (n_components={pca.n_components_})")

    with open(args.ridge_model, 'rb') as f:
        ridge = pickle.load(f)
    log.info(f"  ✓ Ridge model loaded (alpha={ridge.alpha})")

    # Load representations
    log.info(f"\nLoading cell group representations...")
    with open(args.representations_main) as f:
        main_reps = json.load(f)
    log.info(f"  ✓ Main cells: {len(main_reps)} cells")

    with open(args.representations_moderator) as f:
        moderator_reps = json.load(f)
    log.info(f"  ✓ Moderator cells: {len(moderator_reps)} cells")

    # Generate predictions for main cells
    log.info(f"\n{'='*78}")
    log.info("Generating Predictions for Main Cell Groups")
    log.info(f"{'='*78}")

    main_predictions = {}
    for cell_idx, (cell_key, rep_vector) in enumerate(sorted(main_reps.items())):
        if (cell_idx + 1) % 50 == 0:
            log.info(f"  Main cell {cell_idx + 1}/{len(main_reps)}: {cell_key}")

        # Ensure rep_vector is a numpy array
        rep_array = np.array(rep_vector, dtype=np.float32).reshape(1, -1)

        # Transform through PCA
        rep_pca = pca.transform(rep_array)

        # Predict through Ridge
        pred = ridge.predict(rep_pca)[0]

        main_predictions[cell_key] = float(pred)

    log.info(f"Generated predictions for {len(main_predictions)} main cell groups")

    # Generate predictions for moderator cells
    log.info(f"\n{'='*78}")
    log.info("Generating Predictions for Moderator Cell Groups")
    log.info(f"{'='*78}")

    moderator_predictions = {}
    for cell_idx, (cell_key, rep_vector) in enumerate(sorted(moderator_reps.items())):
        if (cell_idx + 1) % 500 == 0:
            log.info(f"  Moderator cell {cell_idx + 1}/{len(moderator_reps)}: {cell_key}")

        # Ensure rep_vector is a numpy array
        rep_array = np.array(rep_vector, dtype=np.float32).reshape(1, -1)

        # Transform through PCA
        rep_pca = pca.transform(rep_array)

        # Predict through Ridge
        pred = ridge.predict(rep_pca)[0]

        moderator_predictions[cell_key] = float(pred)

    log.info(f"Generated predictions for {len(moderator_predictions)} moderator cell groups")

    # Write main cell predictions to CSV
    log.info(f"\n{'='*78}")
    log.info("Writing Output Files")
    log.info(f"{'='*78}")

    main_csv_file = output_dir / "T2_primary_v1_cells_main.csv"
    with open(main_csv_file, 'w') as f:
        f.write("condition,outcome,mean\n")
        for cell_key in sorted(main_predictions.keys()):
            parts = cell_key.split("__")
            condition = parts[0]
            outcome = parts[1]
            mean = main_predictions[cell_key]
            f.write(f"{condition},{outcome},{mean}\n")
    log.info(f"✓ Main predictions written to {main_csv_file}")

    # Write moderator cell predictions to CSV
    moderator_csv_file = output_dir / "T2_primary_v1_cells_moderator.csv"
    with open(moderator_csv_file, 'w') as f:
        f.write("condition,moderator,moderator_level,outcome,mean\n")
        for cell_key in sorted(moderator_predictions.keys()):
            parts = cell_key.split("__")
            condition = parts[0]
            moderator = parts[1]
            level = parts[2]
            if "," in level:
                level = f'"{level}"'
            outcome = parts[3]
            mean = moderator_predictions[cell_key]
            f.write(f"{condition},{moderator},{level},{outcome},{mean}\n")
    log.info(f"✓ Moderator predictions written to {moderator_csv_file}")

    # Generate prediction statistics and report
    log.info(f"\n{'='*78}")
    log.info("Prediction Statistics")
    log.info(f"{'='*78}")

    main_preds_array = np.array(list(main_predictions.values()))
    moderator_preds_array = np.array(list(moderator_predictions.values()))

    log.info(f"\nMain cell predictions:")
    log.info(f"  Mean: {np.mean(main_preds_array):.6f}")
    log.info(f"  Std:  {np.std(main_preds_array):.6f}")
    log.info(f"  Min:  {np.min(main_preds_array):.6f}")
    log.info(f"  Max:  {np.max(main_preds_array):.6f}")

    log.info(f"\nModerator cell predictions:")
    log.info(f"  Mean: {np.mean(moderator_preds_array):.6f}")
    log.info(f"  Std:  {np.std(moderator_preds_array):.6f}")
    log.info(f"  Min:  {np.min(moderator_preds_array):.6f}")
    log.info(f"  Max:  {np.max(moderator_preds_array):.6f}")

    # Per-condition breakdown for main cells
    log.info(f"\n{'='*78}")
    log.info("Per-Condition Breakdown (Main Cells)")
    log.info(f"{'='*78}")

    by_condition = defaultdict(list)
    for cell_key, pred in main_predictions.items():
        condition = cell_key.split("__")[0]
        by_condition[condition].append(pred)

    for condition in sorted(by_condition.keys()):
        preds = np.array(by_condition[condition])
        log.info(f"\n{condition}:")
        log.info(f"  n=       {len(preds)}")
        log.info(f"  mean:    {np.mean(preds):.6f}")
        log.info(f"  std:     {np.std(preds):.6f}")

    # Per-condition-moderator breakdown
    log.info(f"\n{'='*78}")
    log.info("Per-Moderator Breakdown (Sample)")
    log.info(f"{'='*78}")

    by_moderator = defaultdict(list)
    for cell_key, pred in moderator_predictions.items():
        parts = cell_key.split("__")
        condition_moderator = f"{parts[0]}__{parts[1]}"
        by_moderator[condition_moderator].append(pred)

    sample_keys = sorted(by_moderator.keys())[:10]
    for key in sample_keys:
        preds = np.array(by_moderator[key])
        log.info(f"\n{key}:")
        log.info(f"  n=   {len(preds)}")
        log.info(f"  mean: {np.mean(preds):.6f}")

    # Save detailed report
    report = {
        'timestamp': str(__import__('time').time()),
        'model': 'Ridge + PCA (quintuple-source trained)',
        'pca_components': int(pca.n_components_),
        'ridge_alpha': float(ridge.alpha),
        'main_cells': {
            'n': len(main_predictions),
            'mean': float(np.mean(main_preds_array)),
            'std': float(np.std(main_preds_array)),
            'min': float(np.min(main_preds_array)),
            'max': float(np.max(main_preds_array)),
        },
        'moderator_cells': {
            'n': len(moderator_predictions),
            'mean': float(np.mean(moderator_preds_array)),
            'std': float(np.std(moderator_preds_array)),
            'min': float(np.min(moderator_preds_array)),
            'max': float(np.max(moderator_preds_array)),
        },
        'by_condition': {
            condition: {
                'n': len(preds),
                'mean': float(np.mean(preds)),
                'std': float(np.std(preds)),
            }
            for condition, preds in (
                (cond, np.array(by_condition[cond]))
                for cond in sorted(by_condition.keys())
            )
        },
    }

    report_file = output_dir / "prediction_report_quintuple.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    log.info(f"\n✓ Report saved to {report_file}")

    # Summary
    log.info(f"\n{'='*78}")
    log.info("PREDICTION COMPLETE")
    log.info(f"{'='*78}")
    log.info(f"\nOutput files:")
    log.info(f"  Main predictions:      {main_csv_file}")
    log.info(f"  Moderator predictions: {moderator_csv_file}")
    log.info(f"  Report:                {report_file}")
    log.info(f"\nNext steps:")
    log.info(f"  1. Compare predictions with /predictions/example_T2_primary_v1_cells_*.csv")
    log.info(f"  2. Evaluate prediction accuracy if ground truth is available")
    log.info(f"  3. Analyze per-condition and per-outcome performance")


if __name__ == "__main__":
    main()
