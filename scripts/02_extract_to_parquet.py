from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from faers_sglt2_dka.io import extract_and_convert_quarters


def main():
    parser = argparse.ArgumentParser(description="Extract FAERS ZIPs and convert ASCII tables to Parquet.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_and_convert_quarters(args.raw_dir, args.interim_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
