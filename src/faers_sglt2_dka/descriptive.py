"""Descriptive statistics for FAERS SGLT2-DKA study.

Generates:
- Table 1: Demographics and report characteristics (overall / DKA / non-DKA)
- Table 2: DKA report distribution by individual SGLT2 inhibitor
- Figure 1: Data screening flowchart (CONSORT-like)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def generate_table1(all_cases: pd.DataFrame, label_col: str = "label_target_event") -> pd.DataFrame:
    """Generate Table 1: Demographics and report characteristics stratified by DKA status.

    Only includes SGLT2-inhibitor reports (has_study_drug_any == 1).
    """
    df = all_cases[all_cases["has_study_drug_any"].eq(1)].copy()
    dka = df[df[label_col].eq(1)]
    non_dka = df[df[label_col].eq(0)]

    rows = []

    def add_row(label: str, overall_val, dka_val, non_dka_val):
        rows.append({
            "Characteristic": label,
            "Overall (N={})".format(len(df)): overall_val,
            "DKA (n={})".format(len(dka)): dka_val,
            "Non-DKA (n={})".format(len(non_dka)): non_dka_val,
        })

    # N
    add_row("N (reports)", len(df), len(dka), len(non_dka))

    # Age
    if "age_num" in df.columns:
        for subset, name in [(df, "Overall"), (dka, "DKA"), (non_dka, "Non-DKA")]:
            pass  # computed below
        age_all = f"{df['age_num'].median():.0f} ({df['age_num'].quantile(0.25):.0f}–{df['age_num'].quantile(0.75):.0f})"
        age_dka = f"{dka['age_num'].median():.0f} ({dka['age_num'].quantile(0.25):.0f}–{dka['age_num'].quantile(0.75):.0f})"
        age_nondka = f"{non_dka['age_num'].median():.0f} ({non_dka['age_num'].quantile(0.25):.0f}–{non_dka['age_num'].quantile(0.75):.0f})"
        add_row("Age, median (IQR)", age_all, age_dka, age_nondka)

    # Sex
    if "sex" in df.columns:
        for val in ["F", "M"]:
            n_all = (df["sex"] == val).sum()
            n_dka = (dka["sex"] == val).sum()
            n_nondka = (non_dka["sex"] == val).sum()
            pct_all = n_all / len(df) * 100 if len(df) > 0 else 0
            pct_dka = n_dka / len(dka) * 100 if len(dka) > 0 else 0
            pct_nondka = n_nondka / len(non_dka) * 100 if len(non_dka) > 0 else 0
            label = "Female, n (%)" if val == "F" else "Male, n (%)"
            add_row(label,
                    f"{n_all} ({pct_all:.1f}%)",
                    f"{n_dka} ({pct_dka:.1f}%)",
                    f"{n_nondka} ({pct_nondka:.1f}%)")

    # Reporter country (US vs non-US)
    if "occr_country" in df.columns:
        n_us_all = df["occr_country"].str.upper().eq("US").sum()
        n_us_dka = dka["occr_country"].str.upper().eq("US").sum()
        n_us_nondka = non_dka["occr_country"].str.upper().eq("US").sum()
        add_row("US report, n (%)",
                f"{n_us_all} ({n_us_all/len(df)*100:.1f}%)",
                f"{n_us_dka} ({n_us_dka/len(dka)*100:.1f}%)" if len(dka) > 0 else "0 (0%)",
                f"{n_us_nondka} ({n_us_nondka/len(non_dka)*100:.1f}%)" if len(non_dka) > 0 else "0 (0%)")

    # Reporter type
    if "reporter_type" in df.columns:
        for rt in df["reporter_type"].value_counts().head(5).index:
            n_all = (df["reporter_type"] == rt).sum()
            n_dka = (dka["reporter_type"] == rt).sum()
            n_nondka = (non_dka["reporter_type"] == rt).sum()
            add_row(f"Reporter: {rt}, n (%)",
                    f"{n_all} ({n_all/len(df)*100:.1f}%)",
                    f"{n_dka} ({n_dka/len(dka)*100:.1f}%)" if len(dka) > 0 else "0 (0%)",
                    f"{n_nondka} ({n_nondka/len(non_dka)*100:.1f}%)" if len(non_dka) > 0 else "0 (0%)")

    # Study drug role (Primary Suspect)
    if "study_drug_role" in df.columns:
        n_ps_all = df["study_drug_role"].str.upper().eq("PS").sum()
        n_ps_dka = dka["study_drug_role"].str.upper().eq("PS").sum()
        n_ps_nondka = non_dka["study_drug_role"].str.upper().eq("PS").sum()
        add_row("Primary suspect, n (%)",
                f"{n_ps_all} ({n_ps_all/len(df)*100:.1f}%)",
                f"{n_ps_dka} ({n_ps_dka/len(dka)*100:.1f}%)" if len(dka) > 0 else "0 (0%)",
                f"{n_ps_nondka} ({n_ps_nondka/len(non_dka)*100:.1f}%)" if len(non_dka) > 0 else "0 (0%)")

    # Concomitant drugs
    for col in sorted(c for c in df.columns if c.startswith("concomitant_")):
        n_all = df[col].sum()
        n_dka = dka[col].sum()
        n_nondka = non_dka[col].sum()
        drug_name = col.replace("concomitant_", "").replace("_", " ")
        add_row(f"Concomitant {drug_name}, n (%)",
                f"{int(n_all)} ({n_all/len(df)*100:.1f}%)",
                f"{int(n_dka)} ({n_dka/len(dka)*100:.1f}%)" if len(dka) > 0 else "0 (0%)",
                f"{int(n_nondka)} ({n_nondka/len(non_dka)*100:.1f}%)" if len(non_dka) > 0 else "0 (0%)")

    # Drug count
    if "drug_count" in df.columns:
        add_row("Concomitant drugs, median (IQR)",
                f"{df['drug_count'].median():.0f} ({df['drug_count'].quantile(0.25):.0f}–{df['drug_count'].quantile(0.75):.0f})",
                f"{dka['drug_count'].median():.0f} ({dka['drug_count'].quantile(0.25):.0f}–{dka['drug_count'].quantile(0.75):.0f})",
                f"{non_dka['drug_count'].median():.0f} ({non_dka['drug_count'].quantile(0.25):.0f}–{non_dka['drug_count'].quantile(0.75):.0f})")

    # Serious outcomes (descriptive only, NOT model features)
    for col in ["outcome_death", "outcome_hospitalization", "outcome_life_threatening"]:
        if col in df.columns:
            n_all = df[col].sum()
            n_dka = dka[col].sum()
            n_nondka = non_dka[col].sum()
            label = col.replace("outcome_", "").replace("_", " ").title()
            add_row(f"{label}, n (%)",
                    f"{int(n_all)} ({n_all/len(df)*100:.1f}%)",
                    f"{int(n_dka)} ({n_dka/len(dka)*100:.1f}%)" if len(dka) > 0 else "0 (0%)",
                    f"{int(n_nondka)} ({n_nondka/len(non_dka)*100:.1f}%)" if len(non_dka) > 0 else "0 (0%)")

    return pd.DataFrame(rows)


def generate_table2(all_cases: pd.DataFrame, label_col: str = "label_target_event") -> pd.DataFrame:
    """Generate Table 2: DKA report distribution by individual SGLT2 inhibitor."""
    df = all_cases[all_cases["has_study_drug_any"].eq(1)].copy()
    drug_cols = [c for c in df.columns if c.startswith("has_") and c not in {"has_study_drug_any"}]

    rows = []
    for col in drug_cols:
        drug_name = col.replace("has_", "").replace("_", " ")
        drug_reports = df[df[col].eq(1)]
        n_total = len(drug_reports)
        n_dka = int(drug_reports[label_col].sum())
        pct_dka = n_dka / n_total * 100 if n_total > 0 else 0

        # Serious outcomes within this drug's DKA reports
        dka_reports = drug_reports[drug_reports[label_col].eq(1)]
        n_death = int(dka_reports.get("outcome_death", pd.Series(dtype=int)).sum()) if "outcome_death" in dka_reports.columns else 0
        n_hosp = int(dka_reports.get("outcome_hospitalization", pd.Series(dtype=int)).sum()) if "outcome_hospitalization" in dka_reports.columns else 0

        rows.append({
            "SGLT2 inhibitor": drug_name,
            "Total reports": n_total,
            "DKA reports": n_dka,
            "DKA report %": f"{pct_dka:.1f}%",
            "Hospitalization in DKA": n_hosp,
            "Death in DKA": n_death,
        })

    return pd.DataFrame(rows)


def generate_flowchart(screening: dict, out_path: str | Path) -> None:
    """Generate a data screening flowchart (Figure 1) similar to PRISMA/CONSORT.

    The flowchart shows:
    Total FAERS reports → After dedup → SGLT2 reports → DKA vs non-DKA
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Data Screening Flowchart", fontsize=14, fontweight="bold", pad=20)

    def draw_box(x, y, w, h, text, facecolor="#E8F0FE", fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.2", facecolor=facecolor, edgecolor="black", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True,
                multialignment="center")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Step 1: Total reports
    n_total = screening.get("total_after_dedup", "?")
    draw_box(5, 11, 5, 0.8, f"FAERS reports after deduplication\n(N = {n_total:,})")

    # Arrow down
    draw_arrow(5, 10.6, 5, 10.0)

    # Step 2: With features
    n_feat = screening.get("total_with_features", "?")
    draw_box(5, 9.6, 5, 0.8, f"Reports with complete feature extraction\n(N = {n_feat:,})")

    # Arrow down
    draw_arrow(5, 9.2, 5, 8.6)

    # Step 3: SGLT2 reports
    n_sglt2 = screening.get("sglt2_reports", "?")
    draw_box(5, 8.2, 5, 0.8, f"SGLT2 inhibitor-related reports\n(n = {n_sglt2:,})",
             facecolor="#FFF3E0")

    # Arrow splits
    draw_arrow(5, 7.8, 3, 7.2)
    draw_arrow(5, 7.8, 7, 7.2)

    # Step 4a: DKA reports
    n_dka = screening.get("dka_reports_in_sglt2", "?")
    draw_box(3, 6.8, 3.5, 0.8, f"DKA-related reports\n(n = {n_dka:,})",
             facecolor="#FFCDD2")

    # Step 4b: Non-DKA reports
    n_nondka = screening.get("non_dka_reports_in_sglt2", "?")
    draw_box(7, 6.8, 3.5, 0.8, f"Non-DKA reports\n(n = {n_nondka:,})",
             facecolor="#C8E6C9")

    # Step 5: Model dataset
    draw_arrow(3, 6.4, 3, 5.8)
    draw_arrow(7, 6.4, 7, 5.8)
    draw_arrow(3, 5.4, 5, 4.8)
    draw_arrow(7, 5.4, 5, 4.8)
    draw_box(5, 4.4, 5, 0.8,
             f"Model dataset (SGLT2 reports only)\n(n = {n_sglt2:,})\n"
             f"Positive: {n_dka:,} | Negative: {n_nondka:,}",
             facecolor="#E1BEE7")

    # Step 6: Signal detection & ML
    draw_arrow(5, 4.0, 5, 3.4)
    draw_box(5, 3.0, 5, 0.8,
             "Signal detection (ROR/PRR)\n+ Machine learning + SHAP",
             facecolor="#B2EBF2")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_descriptive(all_cases: pd.DataFrame, model_dataset: pd.DataFrame,
                    out_dir: str | Path, screening: dict | None = None,
                    label_col: str = "label_target_event") -> None:
    """Run all descriptive analyses and save outputs."""
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Table 1
    table1 = generate_table1(all_cases, label_col=label_col)
    table1.to_csv(out_dir / "tables" / "table1_demographics.csv", index=False)
    print("[descriptive] Table 1 saved")

    # Table 2
    table2 = generate_table2(all_cases, label_col=label_col)
    table2.to_csv(out_dir / "tables" / "table2_drug_distribution.csv", index=False)
    print("[descriptive] Table 2 saved")

    # Figure 1: Flowchart
    if screening is not None:
        generate_flowchart(screening, out_dir / "figures" / "figure1_flowchart.png")
        print("[descriptive] Figure 1 (flowchart) saved")
