from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from faers_sglt2_dka.modeling import train_models


def main():
    parser = argparse.ArgumentParser(description="Train ML models for target-event report identification.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility (default: 42)")
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.processed_dir) / "model_dataset.parquet")
    metrics, _, test_predictions = train_models(df, args.out_dir, label_col="label_target_event",
                                              test_size=args.test_size, random_state=args.random_state)
    print("\nModel Training Results:")
    print("=" * 60)
    print(metrics.to_string(index=False))
    print(f"\nBest model: {metrics.iloc[0]['model']} (PR-AUC: {metrics.iloc[0]['PR_AUC']:.4f})")
    if test_predictions is not None:
        print(f"\nTest predictions saved to {args.out_dir}/tables/test_predictions.csv")


if __name__ == "__main__":
    main()
