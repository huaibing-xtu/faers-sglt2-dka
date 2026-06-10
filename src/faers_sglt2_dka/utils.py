from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(value) -> str:
    """Normalize FAERS free text for robust matching."""
    if pd.isna(value):
        return ""
    x = str(value).upper()
    x = x.replace("\u00a0", " ")
    x = re.sub(r"[^A-Z0-9/+\- ]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_quarter(q: str) -> str:
    q = str(q).upper().strip()
    m = re.fullmatch(r"(20\d{2})Q([1-4])", q)
    if not m:
        raise ValueError(f"Invalid quarter format: {q}; expected e.g. 2015Q1")
    return f"{m.group(1)}Q{m.group(2)}"


def quarter_range(start: str, end: str) -> list[str]:
    start = normalize_quarter(start)
    end = normalize_quarter(end)
    sy, sq = int(start[:4]), int(start[-1])
    ey, eq = int(end[:4]), int(end[-1])
    out = []
    y, q = sy, sq
    while (y < ey) or (y == ey and q <= eq):
        out.append(f"{y}Q{q}")
        q += 1
        if q == 5:
            y += 1
            q = 1
    return out


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def select_columns_if_exist(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    present = [c for c in columns if c in df.columns]
    return df[present].copy()


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_faers_date(s: pd.Series) -> pd.Series:
    """Parse YYYYMMDD FAERS dates. Invalid partial dates become NaT."""
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
