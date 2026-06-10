from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import numpy as np
import pandas as pd

from .utils import ensure_dir, load_yaml, normalize_text, parse_faers_date


def available_quarter_dirs(interim_dir: str | Path) -> list[Path]:
    interim_dir = Path(interim_dir)
    return sorted([p for p in interim_dir.iterdir() if p.is_dir() and p.name[:4].isdigit() and "Q" in p.name])


def read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def concat_table(interim_dir: str | Path, table: str, columns: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for qdir in available_quarter_dirs(interim_dir):
        p = qdir / f"{table}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            if "quarter" in df.columns and "quarter" not in keep:
                keep.append("quarter")
            df = df[keep]
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No table {table}.parquet found in {interim_dir}")
    return pd.concat(parts, ignore_index=True)


def build_deduplicated_demo(interim_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    """Global deduplication across all quarters. Rule: for each caseid, keep latest fda_dt; if tied, keep max primaryid."""
    processed_dir = ensure_dir(processed_dir)
    demo_cols = [
        "primaryid", "caseid", "caseversion", "fda_dt", "event_dt", "init_fda_dt",
        "age", "age_cod", "age_grp", "sex", "wt", "wt_cod", "occp_cod",
        "reporter_country", "occr_country", "quarter",
    ]
    demo = concat_table(interim_dir, "demo", columns=demo_cols)
    demo.columns = [c.lower() for c in demo.columns]
    if "primaryid" not in demo.columns:
        raise KeyError("DEMO table must contain primaryid")
    if "caseid" not in demo.columns:
        demo["caseid"] = demo["primaryid"]

    for c in ["primaryid", "caseid"]:
        demo[c] = demo[c].astype(str).str.strip()

    date_col = "fda_dt" if "fda_dt" in demo.columns else None
    if date_col is None:
        demo["fda_dt"] = pd.NA
    demo["fda_dt_parsed"] = parse_faers_date(demo["fda_dt"].fillna(""))
    if "init_fda_dt" in demo.columns:
        demo["init_fda_dt_parsed"] = parse_faers_date(demo["init_fda_dt"].fillna(""))
        demo["fda_dt_parsed"] = demo["fda_dt_parsed"].fillna(demo["init_fda_dt_parsed"])

    demo["primaryid_num"] = pd.to_numeric(demo["primaryid"], errors="coerce")
    demo = demo.sort_values(["caseid", "fda_dt_parsed", "primaryid_num"])
    dedup = demo.drop_duplicates("caseid", keep="last").drop(
        columns=[c for c in ["primaryid_num", "init_fda_dt_parsed"] if c in demo.columns]
    )

    dedup.to_parquet(processed_dir / "dedup_demo.parquet", index=False)
    pd.DataFrame({"primaryid": dedup["primaryid"].astype(str)}).to_parquet(
        processed_dir / "dedup_primaryids.parquet", index=False
    )
    return dedup


def make_term_matcher(term_map: dict[str, list[str]]):
    """Word-level term matcher."""
    normalized = {}
    for key, terms in term_map.items():
        normalized[key] = [normalize_text(t).split() for t in terms]

    def identify(value: str) -> str | None:
        text = normalize_text(value)
        if not text:
            return None
        tokens = text.split()
        for key, term_token_lists in normalized.items():
            for term_tokens in term_token_lists:
                if all(tok in tokens for tok in term_tokens if tok):
                    return key
        return None

    return identify


def _make_flat_terms(terms: Iterable[str]) -> str:
    """Build a simple OR regex pattern from a list of terms for str.contains."""
    parts = []
    for term in terms:
        toks = normalize_text(term).split()
        if toks:
            parts.append("|".join(toks))
    return "|".join(parts) if parts else "^(?!x)"


def _make_drug_patterns(term_map: dict[str, list[str]]) -> dict[str, str]:
    """Build per-drug OR-regex patterns for str.contains."""
    out = {}
    for key, terms in term_map.items():
        tokens = set()
        for term in terms:
            tokens.update(t for t in normalize_text(term).split() if t)
        out[key] = "|".join(sorted(tokens))
    return out


def _match_drug_vectorized(drug_text: pd.Series, patterns: dict[str, str]) -> pd.DataFrame:
    """Vectorized drug identification via str.contains with OR regex patterns."""
    result = pd.DataFrame(index=drug_text.index)
    for drug_key, pattern in patterns.items():
        result[drug_key] = drug_text.str.contains(pattern, regex=True, na=False)
    return result


def build_reaction_labels(interim_dir: str | Path, processed_dir: str | Path, cfg: dict) -> pd.DataFrame:
    processed_dir = ensure_dir(processed_dir)
    keep_ids = set(
        pd.read_parquet(Path(processed_dir) / "dedup_primaryids.parquet")["primaryid"].astype(str)
    )
    event_terms_key = cfg.get("target_event_terms_key", "target_event_terms")
    event_terms = {normalize_text(t) for t in cfg[event_terms_key]}

    labels = []
    reaction_counts = []
    for qdir in available_quarter_dirs(interim_dir):
        df = read_parquet_if_exists(qdir / "reac.parquet")
        if df is None:
            continue
        df.columns = [c.lower() for c in df.columns]
        if "primaryid" not in df.columns or "pt" not in df.columns:
            continue
        df["primaryid"] = df["primaryid"].astype(str).str.strip()
        df = df[df["primaryid"].isin(keep_ids)].copy()
        df["pt_norm"] = df["pt"].map(normalize_text)
        df["is_target_event"] = df["pt_norm"].isin(event_terms)
        labels.append(df.groupby("primaryid", as_index=False)["is_target_event"].max())
        reaction_counts.append(
            df.groupby("primaryid", as_index=False).agg(reaction_count=("pt_norm", "nunique"))
        )

    if labels:
        label = pd.concat(labels).groupby("primaryid", as_index=False)["is_target_event"].max()
    else:
        label = pd.DataFrame(columns=["primaryid", "is_target_event"])
    label["label_target_event"] = label["is_target_event"].fillna(False).astype(int)
    label = label[["primaryid", "label_target_event"]]

    if reaction_counts:
        rc = pd.concat(reaction_counts).groupby("primaryid", as_index=False)["reaction_count"].sum()
        label = label.merge(rc, on="primaryid", how="outer")
    label["label_target_event"] = label["label_target_event"].fillna(0).astype(int)
    label["reaction_count"] = label.get("reaction_count", 0).fillna(0).astype(int)
    label.to_parquet(processed_dir / "reaction_labels.parquet", index=False)
    return label


def build_drug_features(interim_dir: str | Path, processed_dir: str | Path, cfg: dict) -> pd.DataFrame:
    """Fully vectorized drug feature building.

    Two key optimizations:
    1. Drug matching via regex str.contains on a Series (C-level, ~10x faster than Python map)
    2. Concomitant detection via same str.contains (vectorized per-row)
    3. Aggregation via groupby().agg("max") (no Python loops)
    """
    processed_dir = ensure_dir(processed_dir)
    keep_ids = set(
        pd.read_parquet(Path(processed_dir) / "dedup_primaryids.parquet")["primaryid"].astype(str)
    )
    study_drug_keys = list(cfg["study_drugs"].keys())
    concomitant_groups = cfg.get("concomitant_drug_groups", {})

    # Pre-build all regex patterns once (done outside the quarter loop)
    drug_patterns = _make_drug_patterns(cfg["study_drugs"])
    concomitant_patterns = {
        grp_name: _make_flat_terms(terms) for grp_name, terms in concomitant_groups.items()
    }

    parts = []
    for qdir in available_quarter_dirs(interim_dir):
        df = read_parquet_if_exists(qdir / "drug.parquet")
        if df is None:
            continue
        df.columns = [c.lower() for c in df.columns]
        if "primaryid" not in df.columns:
            continue
        df["primaryid"] = df["primaryid"].astype(str).str.strip()
        df = df[df["primaryid"].isin(keep_ids)].copy()
        if df.empty:
            continue

        for col in ["drugname", "prod_ai", "role_cod", "route"]:
            if col not in df.columns:
                df[col] = pd.NA

        df["drug_text"] = (
            df["drugname"].fillna("") + " " + df["prod_ai"].fillna("")
        ).map(normalize_text)

        # --- Vectorized drug matching via str.contains (C-level speed) ---
        drug_match = _match_drug_vectorized(df["drug_text"], drug_patterns)
        # study_drug: first drug that matches, or NaN
        df["study_drug"] = drug_match.idxmax(axis=1).where(drug_match.any(axis=1))

        # Per-drug binary flags from the match matrix
        for dk in study_drug_keys:
            df[f"_has_{dk}"] = drug_match[dk].astype(np.int8) if dk in drug_match.columns else np.int8(0)

        # --- Concomitant detection via str.contains (vectorized) ---
        for grp_name, pattern in concomitant_patterns.items():
            df[f"_concom_{grp_name}"] = (
                df["drug_text"].str.contains(pattern, regex=True, na=False)
            ).astype(np.int8)

        cols = ["primaryid", "drug_text", "study_drug", "role_cod", "route"]
        cols += [c for c in df.columns if c.startswith("_has_") or c.startswith("_concom_")]
        parts.append(df[cols])

    if not parts:
        raise RuntimeError("No DRUG rows found after filtering to deduplicated primary IDs")

    drug = pd.concat(parts, ignore_index=True)
    del parts

    # --- Phase 2: study_drug_main/role/route per primaryid (PS priority) ---
    ps_mask = drug["role_cod"].astype(str).str.upper().eq("PS")
    ps_agg = (
        drug[ps_mask].groupby("primaryid", as_index=False)
        .agg(
            study_drug_main_ps=("study_drug", "first"),
            study_drug_role_ps=("role_cod", "first"),
            study_drug_route_ps=("route", "first"),
        )
    )
    all_agg = (
        drug.groupby("primaryid", as_index=False)
        .agg(
            study_drug_main_all=("study_drug", "first"),
            study_drug_role_all=("role_cod", "first"),
            study_drug_route_all=("route", "first"),
        )
    )
    main = all_agg.merge(ps_agg, on="primaryid", how="left")
    main["study_drug_main"] = main["study_drug_main_ps"].fillna(main["study_drug_main_all"])
    main["study_drug_role"] = main["study_drug_role_ps"].fillna(main["study_drug_role_all"])
    main["study_drug_route"] = main["study_drug_route_ps"].fillna(main["study_drug_route_all"])
    main = main[["primaryid", "study_drug_main", "study_drug_role", "study_drug_route"]]

    # --- Phase 3: Aggregate binary flags via groupby().agg("max") ---
    has_cols = [c for c in drug.columns if c.startswith("_has_")]
    con_cols = [c for c in drug.columns if c.startswith("_concom_")]

    # Pre-compute drug_count using nunique (vectorized C-level, much faster than Python lambda)
    drug_count = drug.groupby("primaryid", sort=False)["drug_text"].nunique().reset_index(name="drug_count")

    num_agg = {c: "max" for c in has_cols + con_cols}
    agg = drug.groupby("primaryid", sort=False).agg(num_agg).reset_index()
    agg = agg.merge(drug_count, on="primaryid", how="left")

    rename = {}
    for c in has_cols:
        rename[c] = c.replace("_has_", "has_")
    for c in con_cols:
        rename[c] = c.replace("_concom_", "concomitant_")
    agg = agg.rename(columns=rename)

    any_col = "has_study_drug_any"
    has_any_cols = [c for c in agg.columns if c.startswith("has_") and c != any_col]
    agg.insert(1, any_col, agg[has_any_cols].max(axis=1).clip(upper=1).astype(np.int8))

    agg = agg.merge(main, on="primaryid", how="left")
    agg.to_parquet(processed_dir / "drug_features.parquet", index=False)
    return agg


def build_indication_features(interim_dir: str | Path, processed_dir: str | Path, cfg: dict) -> pd.DataFrame:
    """Vectorized indication feature building."""
    processed_dir = ensure_dir(processed_dir)
    keep_ids = set(
        pd.read_parquet(Path(processed_dir) / "dedup_primaryids.parquet")["primaryid"].astype(str)
    )
    indication_groups = cfg.get("indication_groups", {})

    parts = []
    for qdir in available_quarter_dirs(interim_dir):
        df = read_parquet_if_exists(qdir / "indi.parquet")
        if df is None:
            continue
        df.columns = [c.lower() for c in df.columns]
        if "primaryid" not in df.columns:
            continue
        if "indi_pt" not in df.columns:
            possible = [c for c in df.columns if "indi" in c and "pt" in c]
            if possible:
                df = df.rename(columns={possible[0]: "indi_pt"})
            else:
                continue
        df["primaryid"] = df["primaryid"].astype(str).str.strip()
        df = df[df["primaryid"].isin(keep_ids)].copy()
        df["indi_norm"] = df["indi_pt"].map(normalize_text)

        for grp_name, terms in indication_groups.items():
            pattern = _make_flat_terms(terms)
            df[f"_ind_{grp_name}"] = (
                df["indi_norm"].str.contains(pattern, regex=True, na=False)
            ).astype(np.int8)

        cols = ["primaryid", "indi_norm"] + [c for c in df.columns if c.startswith("_ind_")]
        parts.append(df[cols])

    if not parts:
        return pd.DataFrame({"primaryid": list(keep_ids)})

    indi = pd.concat(parts, ignore_index=True)
    indi_count = indi.groupby("primaryid", sort=False)["indi_norm"].nunique().reset_index(name="indication_count")

    ind_agg_dict = {c: "max" for c in indi.columns if c.startswith("_ind_")}
    agg = indi.groupby("primaryid", sort=False).agg(ind_agg_dict).reset_index()
    agg = agg.merge(indi_count, on="primaryid", how="left")
    agg = agg.rename(columns={c: c.replace("_ind_", "ind_") for c in agg.columns})

    agg.to_parquet(processed_dir / "indication_features.parquet", index=False)
    return agg


def build_outcome_features(interim_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    """Build outcome features (already vectorized)."""
    processed_dir = ensure_dir(processed_dir)
    keep_ids = set(
        pd.read_parquet(Path(processed_dir) / "dedup_primaryids.parquet")["primaryid"].astype(str)
    )
    outcome_map = {
        "DE": "outcome_death",
        "LT": "outcome_life_threatening",
        "HO": "outcome_hospitalization",
        "DS": "outcome_disability",
        "CA": "outcome_congenital_anomaly",
        "RI": "outcome_required_intervention",
        "OT": "outcome_other_serious",
    }
    parts = []
    for qdir in available_quarter_dirs(interim_dir):
        df = read_parquet_if_exists(qdir / "outc.parquet")
        if df is None:
            continue
        df.columns = [c.lower() for c in df.columns]
        if "primaryid" not in df.columns or "outc_cod" not in df.columns:
            continue
        df["primaryid"] = df["primaryid"].astype(str).str.strip()
        df = df[df["primaryid"].isin(keep_ids)].copy()
        df["outc_cod"] = df["outc_cod"].astype(str).str.upper().str.strip()
        parts.append(df[["primaryid", "outc_cod"]])

    if not parts:
        return pd.DataFrame({"primaryid": list(keep_ids)})
    outc = pd.concat(parts, ignore_index=True)
    for code, name in outcome_map.items():
        outc[name] = (outc["outc_cod"] == code).astype(np.int8)
    agg = outc.groupby("primaryid", as_index=False)[list(outcome_map.values())].max()
    agg["any_serious_outcome"] = agg[list(outcome_map.values())].max(axis=1).astype(np.int8)
    agg.to_parquet(processed_dir / "outcome_features.parquet", index=False)
    return agg


def build_reporter_features(interim_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    """Extract reporter type from RPSR table."""
    processed_dir = ensure_dir(processed_dir)
    keep_ids = set(
        pd.read_parquet(Path(processed_dir) / "dedup_primaryids.parquet")["primaryid"].astype(str)
    )
    reporter_map = {
        "MD": "physician",
        "PH": "pharmacist",
        "OT": "other_health_professional",
        "LW": "lawyer",
        "CN": "consumer",
    }
    parts = []
    for qdir in available_quarter_dirs(interim_dir):
        df = read_parquet_if_exists(qdir / "rpsr.parquet")
        if df is None:
            continue
        df.columns = [c.lower() for c in df.columns]
        if "primaryid" not in df.columns or "rpsr_cod" not in df.columns:
            continue
        df["primaryid"] = df["primaryid"].astype(str).str.strip()
        df = df[df["primaryid"].isin(keep_ids)].copy()
        df["rpsr_cod"] = df["rpsr_cod"].astype(str).str.upper().str.strip()
        parts.append(df[["primaryid", "rpsr_cod"]])

    if not parts:
        return pd.DataFrame({"primaryid": list(keep_ids), "reporter_type": "unknown"})
    rpsr = pd.concat(parts, ignore_index=True)
    rpsr["reporter_type"] = rpsr["rpsr_cod"].map(reporter_map).fillna(rpsr["rpsr_cod"].str.lower())
    agg = rpsr.groupby("primaryid", as_index=False)["reporter_type"].first()
    all_ids = pd.DataFrame({"primaryid": list(keep_ids)})
    agg = all_ids.merge(agg, on="primaryid", how="left")
    agg["reporter_type"] = agg["reporter_type"].fillna("unknown")
    agg.to_parquet(processed_dir / "reporter_features.parquet", index=False)
    return agg


def build_analysis_datasets(
    interim_dir: str | Path,
    processed_dir: str | Path,
    config_path: str | Path,
    target_terms_key: str = "target_event_terms",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build analysis datasets from interim FAERS parquet files."""
    cfg = load_yaml(config_path)
    cfg["target_event_terms_key"] = target_terms_key
    processed_dir = ensure_dir(processed_dir)

    screening = {}

    print("1/5: 加载去重后的人口学数据...")
    demo_path = processed_dir / "dedup_demo.parquet"
    if demo_path.exists():
        demo = pd.read_parquet(demo_path)
        print(f"  已加载现有文件: {demo_path} (共 {len(demo):,} 条记录)")
    else:
        demo = build_deduplicated_demo(interim_dir, processed_dir)
        print(f"  构建完成: {len(demo):,} 条去重后记录")
    screening["total_after_dedup"] = len(demo)

    print("2/5: 构建反应标签特征...")
    labels = build_reaction_labels(interim_dir, processed_dir, cfg)
    print(f"  完成: {labels.shape[0]:,} 条记录, {labels['label_target_event'].sum():,} 个DKA标签")

    print("3/5: 构建药物特征...")
    drugs = build_drug_features(interim_dir, processed_dir, cfg)
    print(f"  完成: {drugs.shape[0]:,} 条记录, {drugs['has_study_drug_any'].sum():,} 个SGLT2报告")

    print("4/5: 构建适应症特征...")
    indi = build_indication_features(interim_dir, processed_dir, cfg)
    print(f"  完成: {indi.shape[0]:,} 条记录")

    print("5/5: 构建结果特征和报告者特征...")
    outc = build_outcome_features(interim_dir, processed_dir)
    reporter = build_reporter_features(interim_dir, processed_dir)
    print(f"  完成: {outc.shape[0]:,} 条结果记录, {reporter.shape[0]:,} 条报告者记录")

    demo = demo.copy()
    demo["primaryid"] = demo["primaryid"].astype(str)
    demo["age_num"] = pd.to_numeric(demo.get("age"), errors="coerce")

    # Age unit conversion: convert all age values to years based on age_cod
    # age_cod codes: YR=years, MO=months, WK=weeks, DY=days, HR=hours
    if "age_cod" in demo.columns:
        age_cod_upper = demo["age_cod"].astype(str).str.upper().str.strip()
        # Conversion factors to years
        conversion_factors = {
            "YR": 1.0,
            "MO": 1.0 / 12.0,
            "WK": 1.0 / 52.0,
            "DY": 1.0 / 365.0,
            "HR": 1.0 / (365.0 * 24.0),
        }
        # Apply conversion
        for code, factor in conversion_factors.items():
            mask = age_cod_upper == code
            if mask.any():
                demo.loc[mask, "age_num"] = demo.loc[mask, "age_num"] * factor
        # For unknown age_cod, assume years (most common in FAERS)
        unknown_mask = ~age_cod_upper.isin(conversion_factors.keys()) & demo["age_num"].notna()
        if unknown_mask.any():
            print(f"  警告: {unknown_mask.sum()} 条记录的 age_cod 未知，假设为年")

    # Remove outliers: age > 120 years or age < 0
    age_outliers = ((demo["age_num"] > 120) | (demo["age_num"] < 0)) & demo["age_num"].notna()
    if age_outliers.any():
        print(f"  移除 {age_outliers.sum()} 条年龄异常值 (>120岁 或 <0岁)")
        demo.loc[age_outliers, "age_num"] = np.nan

    demo["report_year"] = pd.to_datetime(demo.get("fda_dt_parsed"), errors="coerce").dt.year
    if demo["report_year"].isna().all() and "quarter" in demo.columns:
        demo["report_year"] = demo["quarter"].astype(str).str[:4].astype(float)

    demo_keep = [
        "primaryid", "caseid", "quarter", "age_num", "age_cod", "age_grp", "sex",
        "wt", "wt_cod", "occp_cod", "reporter_country", "occr_country", "report_year",
    ]
    all_cases = demo[[c for c in demo_keep if c in demo.columns]].copy()
    for part in [labels, drugs, indi, outc, reporter]:
        all_cases = all_cases.merge(part, on="primaryid", how="left")

    all_cases["label_target_event"] = all_cases["label_target_event"].fillna(0).astype(int)
    all_cases["has_study_drug_any"] = all_cases["has_study_drug_any"].fillna(0).astype(int)
    for c in all_cases.columns:
        if (
            c.startswith("has_")
            or c.startswith("concomitant_")
            or c.startswith("ind_")
            or c.startswith("outcome_")
        ):
            all_cases[c] = all_cases[c].fillna(0).astype(int)
    if "any_serious_outcome" in all_cases.columns:
        all_cases["any_serious_outcome"] = all_cases["any_serious_outcome"].fillna(0).astype(int)
    if "reaction_count" in all_cases.columns:
        all_cases["reaction_count"] = all_cases["reaction_count"].fillna(0).astype(int)

    screening["total_with_features"] = len(all_cases)
    screening["sglt2_reports"] = int(all_cases["has_study_drug_any"].sum())
    screening["dka_reports_in_sglt2"] = int(
        all_cases.loc[all_cases["has_study_drug_any"].eq(1), "label_target_event"].sum()
    )
    screening["non_dka_reports_in_sglt2"] = (
        screening["sglt2_reports"] - screening["dka_reports_in_sglt2"]
    )
    screening["dka_reports_all"] = int(all_cases["label_target_event"].sum())

    model_dataset = all_cases[all_cases["has_study_drug_any"].eq(1)].copy()

    all_cases.to_parquet(processed_dir / "all_cases.parquet", index=False)
    model_dataset.to_parquet(processed_dir / "model_dataset.parquet", index=False)

    import json
    with open(processed_dir / "screening_counts.json", "w") as f:
        json.dump(screening, f, indent=2)

    print("\n=== 数据集构建完成 ===")
    print(f"  all_cases.parquet: {len(all_cases):,} 条记录")
    print(f"  model_dataset.parquet: {len(model_dataset):,} 条SGLT2相关记录")
    print(f"  drug_features.parquet: {drugs.shape[0]:,} 条记录")
    print(f"  indication_features.parquet: {indi.shape[0]:,} 条记录")
    print(f"  outcome_features.parquet: {outc.shape[0]:,} 条记录")
    print(f"  reporter_features.parquet: {reporter.shape[0]:,} 条记录")
    print(f"  screening_counts.json: 筛选统计信息")

    return all_cases, model_dataset, screening