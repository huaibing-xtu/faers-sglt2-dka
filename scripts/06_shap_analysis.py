from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from faers_sglt2_dka.explain import run_shap


def main():
    parser = argparse.ArgumentParser(description="Run SHAP explainability analysis on the saved best model.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--max-samples", type=int, default=3000, help="Maximum number of samples for SHAP analysis")
    args = parser.parse_args()

    df = pd.read_parquet(Path(args.processed_dir) / "model_dataset.parquet")
    run_shap(df, args.out_dir, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
