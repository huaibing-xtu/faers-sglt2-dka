from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from faers_sglt2_dka.descriptive import run_descriptive


def main():
    parser = argparse.ArgumentParser(description="Generate descriptive statistics tables and figures.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--label-col", default="label_target_event")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    all_cases = pd.read_parquet(processed / "all_cases.parquet")
    model_dataset = pd.read_parquet(processed / "model_dataset.parquet")

    # Load screening counts
    screening_path = processed / "screening_counts.json"
    screening = None
    if screening_path.exists():
        with open(screening_path) as f:
            screening = json.load(f)
        print("[descriptive] Loaded screening counts:", screening)

    run_descriptive(
        all_cases=all_cases,
        model_dataset=model_dataset,
        out_dir=out,
        screening=screening,
        label_col=args.label_col,
    )
    print("[descriptive] Done. Check outputs/tables/ and outputs/figures/")


if __name__ == "__main__":
    import pandas as pd  # Moved here to avoid early import error
    main()