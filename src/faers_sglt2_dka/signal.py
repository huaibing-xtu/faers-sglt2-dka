from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def compute_ror_prr(a: int, b: int, c: int, d: int) -> dict:
    """Compute ROR, PRR and chi-square for a 2x2 disproportionality table with 95% CI."""
    raw_a, raw_b, raw_c, raw_d = int(a), int(b), int(c), int(d)
    aa, bb, cc, dd = float(a), float(b), float(c), float(d)

    # Handle zero cells with continuity correction
    if min(aa, bb, cc, dd) == 0:
        aa, bb, cc, dd = aa + 0.5, bb + 0.5, cc + 0.5, dd + 0.5

    # Calculate ROR with 95% CI
    ror = (aa * dd) / (bb * cc)
    se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    ror_l = np.exp(np.log(ror) - 1.96 * se)
    ror_u = np.exp(np.log(ror) + 1.96 * se)

    # Calculate PRR with 95% CI
    prr = (aa / (aa + bb)) / (cc / (cc + dd))
    # PRR CI using Wilson score interval
    if aa + bb > 0 and cc + dd > 0:
        p1 = aa / (aa + bb)
        p2 = cc / (cc + dd)
        n1 = aa + bb
        n2 = cc + dd

        # Wilson score interval for PRR
        se_prr = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        prr_l = prr * np.exp(-1.96 * se_prr / prr)
        prr_u = prr * np.exp(1.96 * se_prr / prr)
    else:
        prr_l, prr_u = 0.0, float('inf')

    # Chi-square test
    chi2, p, _, _ = chi2_contingency(np.array([[aa, bb], [cc, dd]]), correction=False)

    # Signal detection criteria
    signal_ror = bool(raw_a >= 3 and ror_l > 1)
    signal_prr = bool(raw_a >= 3 and prr >= 2 and chi2 >= 4)

    return {
        "a": raw_a,
        "b": raw_b,
        "c": raw_c,
        "d": raw_d,
        "ROR": ror,
        "ROR_95CI_low": ror_l,
        "ROR_95CI_high": ror_u,
        "PRR": prr,
        "PRR_95CI_low": prr_l,
        "PRR_95CI_high": prr_u,
        "chi2": chi2,
        "p_value": p,
        "signal_ROR": signal_ror,
        "signal_PRR": signal_prr,
    }


def signal_for_drug_flags(all_cases: pd.DataFrame, drug_flag_cols: list[str], label_col: str = "label_target_event") -> pd.DataFrame:
    rows = []
    y = all_cases[label_col].fillna(0).astype(int)
    for flag in drug_flag_cols:
        x = all_cases[flag].fillna(0).astype(int)
        a = int(((x == 1) & (y == 1)).sum())
        b = int(((x == 1) & (y == 0)).sum())
        c = int(((x == 0) & (y == 1)).sum())
        d = int(((x == 0) & (y == 0)).sum())
        row = {"drug_flag": flag.replace("has_", "")}
        row.update(compute_ror_prr(a, b, c, d))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ROR", ascending=False)


def annual_trend(model_dataset: pd.DataFrame, label_col: str = "label_target_event") -> pd.DataFrame:
    df = model_dataset.copy()
    df["report_year"] = pd.to_numeric(df["report_year"], errors="coerce")
    trend = (
        df.dropna(subset=["report_year"])
        .groupby("report_year")
        .agg(
            total_reports=("primaryid", "count"),
            target_reports=(label_col, "sum"),
        )
        .reset_index()
    )
    trend["target_report_ratio"] = trend["target_reports"] / trend["total_reports"]
    return trend
