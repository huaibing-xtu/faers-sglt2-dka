from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_predictions(pred_file: Path) -> pd.DataFrame:
    """Load prediction data"""
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    df = pd.read_csv(pred_file)
    print(f"Loaded predictions: {len(df)} samples")

    # Check required columns
    required_cols = ['y_true', 'y_prob']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter for the best model if multiple models exist
    if 'model' in df.columns:
        print(f"Found {df['model'].nunique()} models in the file")
        # Group by model and find the best one based on PR-AUC
        best_models = []
        for model_name in df['model'].unique():
            model_data = df[df['model'] == model_name]
            pr_auc = calculate_pr_auc(model_data['y_true'], model_data['y_prob'])
            best_models.append((model_name, pr_auc))

        best_model = max(best_models, key=lambda x: x[1])
        print(f"Using best model: {best_model[0]} (PR-AUC: {best_model[1]:.4f})")
        df = df[df['model'] == best_model[0]].copy()

    return df


def calculate_pr_auc(y_true, y_prob):
    """Calculate PR-AUC"""
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_prob)


def evaluate_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Evaluate performance at a specific threshold"""
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        'threshold': threshold,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Specificity': specificity,
        'Accuracy': accuracy
    }


def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Perform threshold analysis"""
    results = []

    for threshold in thresholds:
        metrics = evaluate_threshold(y_true, y_prob, threshold)
        results.append(metrics)

    return pd.DataFrame(results)


def find_best_thresholds(results_df: pd.DataFrame) -> tuple[float, float]:
    """Find best thresholds"""
    # Find threshold that maximizes F1
    best_f1_idx = results_df['F1'].idxmax()
    best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
    best_f1_value = results_df.loc[best_f1_idx, 'F1']

    # Find threshold with Recall >= 0.8 that maximizes Precision
    recall_thresholds = results_df[results_df['Recall'] >= 0.8]
    if not recall_thresholds.empty:
        best_recall_idx = recall_thresholds['Precision'].idxmax()
        high_recall_threshold = recall_thresholds.loc[best_recall_idx, 'threshold']
        high_recall_precision = recall_thresholds.loc[best_recall_idx, 'Precision']
    else:
        high_recall_threshold = None
        high_recall_precision = None
        print("Warning: No threshold found with Recall >= 0.8")

    return best_f1_threshold, best_f1_value, high_recall_threshold, high_recall_precision


def plot_threshold_curves(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot threshold sensitivity curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Precision, Recall, F1 vs Threshold
    ax1.plot(results_df['threshold'], results_df['Precision'], 'b-', label='Precision', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['Recall'], 'r-', label='Recall', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['F1'], 'g-', label='F1 Score', linewidth=2)

    # Mark best F1 threshold
    best_f1_idx = results_df['F1'].idxmax()
    ax1.plot(results_df.loc[best_f1_idx, 'threshold'], results_df.loc[best_f1_idx, 'F1'],
             'go', markersize=10, label=f'Best F1 ({results_df.loc[best_f1_idx, "F1"]:.3f})')

    # Mark high precision threshold with recall >= 0.8
    recall_thresholds = results_df[results_df['Recall'] >= 0.8]
    if not recall_thresholds.empty:
        best_recall_idx = recall_thresholds['Precision'].idxmax()
        ax1.plot(results_df.loc[best_recall_idx, 'threshold'], results_df.loc[best_recall_idx, 'Precision'],
                 'mo', markersize=10, label=f'High Precision (Recall≥0.8): {results_df.loc[best_recall_idx, "Precision"]:.3f}')

    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, and F1 vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Plot 2: Confusion Matrix components vs Threshold
    ax2.plot(results_df['threshold'], results_df['TP'], 'g-', label='True Positives', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['FP'], 'r-', label='False Positives', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['FN'], 'orange', label='False Negatives', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['TN'], 'b-', label='True Negatives', linewidth=2)

    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Count')
    ax2.set_title('Confusion Matrix Components vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_dir / 'threshold_analysis_curves.png', dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, best_f1_threshold: float,
                  high_recall_threshold: float, out_dir: Path) -> None:
    """Plot Precision-Recall curve with threshold annotations"""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(10, 8))

    # Plot PR curve
    plt.plot(recall, precision, 'b-', label=f'PR Curve (AUC = {pr_auc:.3f})', linewidth=2)

    # Mark thresholds
    if best_f1_threshold is not None:
        idx = np.argmin(np.abs(thresholds - best_f1_threshold))
        if idx < len(precision) - 1:
            plt.plot(recall[idx], precision[idx], 'go', markersize=12,
                    label=f'Best F1 Threshold ({best_f1_threshold:.2f})')

    if high_recall_threshold is not None:
        idx = np.argmin(np.abs(thresholds - high_recall_threshold))
        if idx < len(precision) - 1:
            plt.plot(recall[idx], precision[idx], 'mo', markersize=12,
                    label=f'High Precision Threshold ({high_recall_threshold:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'pr_curve_with_thresholds.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run threshold sensitivity analysis for FAERS SGLT2-DKA model.")
    parser.add_argument("--pred-file", required=True, help="Path to predictions file (output from train_models.py)")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for results")
    args = parser.parse_args()

    pred_file = Path(args.pred_file)
    out_dir = Path(args.out_dir)

    # Create output directories
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FAERS SGLT2-DKA Threshold Sensitivity Analysis")
    print("=" * 60)
    print(f"Predictions file: {pred_file}")
    print(f"Output directory: {out_dir}")
    print("-" * 60)

    try:
        # Load predictions
        df = load_predictions(pred_file)
        y_true = df['y_true'].values
        y_prob = df['y_prob'].values

        # Define thresholds
        thresholds = np.arange(0.05, 1.0, 0.05)
        print(f"Analyzing {len(thresholds)} thresholds from 0.05 to 0.95")

        # Perform threshold analysis
        results_df = threshold_analysis(y_true, y_prob, thresholds)
        results_df.to_csv(tables_dir / "threshold_analysis.csv", index=False)
        print(f"\nThreshold analysis results saved to {tables_dir}/threshold_analysis.csv")

        # Find best thresholds
        best_f1_threshold, best_f1_value, high_recall_threshold, high_recall_precision = find_best_thresholds(results_df)

        # Print results
        print("\nThreshold Analysis Summary:")
        print("=" * 60)
        print(f"Best F1 Score: {best_f1_value:.4f} at threshold = {best_f1_threshold:.2f}")
        print(f"  - Precision: {results_df.loc[results_df['threshold'] == best_f1_threshold, 'Precision'].iloc[0]:.4f}")
        print(f"  - Recall: {results_df.loc[results_df['threshold'] == best_f1_threshold, 'Recall'].iloc[0]:.4f}")

        if high_recall_threshold is not None:
            print(f"High Precision Threshold (Recall≥0.8): {high_recall_precision:.4f} at threshold = {high_recall_threshold:.2f}")
            print(f"  - Recall: {results_df.loc[results_df['threshold'] == high_recall_threshold, 'Recall'].iloc[0]:.4f}")
        else:
            print("No threshold found with Recall >= 0.8")

        # Add best thresholds to results
        results_df.loc[results_df['threshold'] == best_f1_threshold, 'best_f1_threshold'] = True
        if high_recall_threshold is not None:
            results_df.loc[results_df['threshold'] == high_recall_threshold, 'high_recall_threshold'] = True
        results_df.to_csv(tables_dir / "threshold_analysis.csv", index=False)

        # Plot curves
        plot_threshold_curves(results_df, figures_dir)
        print(f"Threshold curves saved to {figures_dir}/threshold_analysis_curves.png")

        # Plot PR curve with threshold annotations
        plot_pr_curve(y_true, y_prob, best_f1_threshold, high_recall_threshold, figures_dir)
        print(f"PR curve saved to {figures_dir}/pr_curve_with_thresholds.png")

        # Print confusion matrices for best thresholds
        print("\nConfusion Matrices:")
        print("=" * 60)

        for threshold, name in [(best_f1_threshold, "Best F1"), (high_recall_threshold, "High Precision")]:
            if threshold is not None:
                y_pred = (y_prob >= threshold).astype(int)
                tn, fp, fn, tp = [[0, 0], [0, 0]]
                cm = [[tn, fp], [fn, tp]]
                # Manually calculate for display
                tp = np.sum((y_true == 1) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))

                print(f"\n{name} Threshold (threshold={threshold:.2f}):")
                print(f"          Predicted Negative    Predicted Positive")
                print(f"Actual Negative     {tn}             {fp}")
                print(f"Actual Positive     {fn}             {tp}")

        print("\n" + "=" * 60)
        print("Threshold sensitivity analysis completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Threshold analysis failed. Please check the error message and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()