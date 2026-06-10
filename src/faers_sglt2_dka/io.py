from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from .utils import ensure_dir

TABLE_PREFIXES = ["DEMO", "DRUG", "REAC", "OUTC", "INDI", "RPSR", "THER"]


def infer_quarter_from_zip(path: str | Path) -> str:
    name = Path(path).name
    m = re.search(r"(20\d{2})[Qq]([1-4])", name)
    if not m:
        raise ValueError(f"Cannot infer quarter from file name: {name}")
    return f"{m.group(1)}Q{m.group(2)}"


def extract_zip(zip_path: str | Path, extract_dir: str | Path, overwrite: bool = False) -> Path:
    zip_path = Path(zip_path)
    quarter = infer_quarter_from_zip(zip_path)
    out_dir = ensure_dir(Path(extract_dir) / quarter)

    marker = out_dir / ".extracted"
    if marker.exists() and not overwrite:
        print(f"[skip] {quarter} already extracted")
        return out_dir

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    marker.touch()
    return out_dir


def find_table_file(extracted_quarter_dir: str | Path, prefix: str) -> Path | None:
    d = Path(extracted_quarter_dir)
    prefix = prefix.upper()
    candidates = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".csv"}:
            if p.name.upper().startswith(prefix):
                candidates.append(p)
    return sorted(candidates)[0] if candidates else None


def read_faers_ascii_table(path: str | Path) -> pd.DataFrame:
    """Read a FAERS ASCII '$'-delimited table and normalize column names to lower case."""
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="$",
        dtype=str,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
        quoting=3,
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def extract_and_convert_quarters(raw_dir: str | Path, interim_dir: str | Path, overwrite: bool = False) -> None:
    """
    Extract each quarterly ZIP and convert the common ASCII tables to Parquet.
    Output layout: interim_dir/YYYYQn/demo.parquet, drug.parquet, ...
    """
    raw_dir = Path(raw_dir)
    interim_dir = ensure_dir(interim_dir)
    extracted_root = ensure_dir(interim_dir / "_extracted")

    zips = sorted(raw_dir.glob("faers_ascii_*.zip"))
    if not zips:
        raise FileNotFoundError(f"No faers_ascii_*.zip found in {raw_dir}")

    for zip_path in zips:
        quarter = infer_quarter_from_zip(zip_path)
        quarter_out = ensure_dir(interim_dir / quarter)
        done_marker = quarter_out / ".converted"
        if done_marker.exists() and not overwrite:
            print(f"[skip] {quarter} already converted")
            continue

        extracted = extract_zip(zip_path, extracted_root, overwrite=overwrite)
        for prefix in TABLE_PREFIXES:
            f = find_table_file(extracted, prefix)
            if f is None:
                print(f"[warn] {quarter}: {prefix} file not found")
                continue
            print(f"[read] {quarter} {prefix}: {f.name}")
            df = read_faers_ascii_table(f)
            df["quarter"] = quarter
            out = quarter_out / f"{prefix.lower()}.parquet"
            df.to_parquet(out, index=False)
            print(f"[write] {out} rows={len(df):,}")
        done_marker.touch()
