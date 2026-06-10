from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from faers_sglt2_dka.download import download_quarters


def main():
    parser = argparse.ArgumentParser(description="Download FAERS ASCII quarterly ZIP files from FDA.")
    parser.add_argument("--start", default="2015Q1")
    parser.add_argument("--end", default="2024Q4")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download_quarters(args.start, args.end, args.raw_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
