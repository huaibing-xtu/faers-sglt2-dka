from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from faers_sglt2_dka.signal import signal_for_drug_flags, annual_trend


def annual_signal_trend(all_cases: pd.DataFrame, drug_flag: str, label_col: str = "label_target_event") -> pd.DataFrame:
    """Calculate annual signal trend for a specific drug."""
    df = all_cases.copy()
    df["report_year"] = pd.to_numeric(df["report_year"], errors="coerce")

    # Filter for the specific drug
    drug_df = df[df[drug_flag] == 1].copy()

    if drug_df.empty:
        return pd.DataFrame()

    trend = (
        drug_df.dropna(subset=["report_year"])
        .groupby("report_year")
        .agg(
            total_reports=("primaryid", "count"),
            target_reports=(label_col, "sum"),
        )
        .reset_index()
    )

    if len(trend) > 1:
        trend["target_report_ratio"] = trend["target_reports"] / trend["total_reports"]

        # Calculate ROR for each year
        a = trend["target_reports"].values
        b = trend["total_reports"] - trend["target_reports"]
        c = trend["target_reports"].shift(1).fillna(0)
        d = trend["total_reports"].shift(1) - trend["target_reports"].shift(1)

        # Apply continuity correction
        a = np.where(a == 0, 0.5, a)
        b = np.where(b == 0, 0.5, b)
        c = np.where(c == 0, 0.5, c)
        d = np.where(d == 0, 0.5, d)

        trend["ROR"] = (a * d) / (b * c)
        trend["ROR_95CI_low"] = np.exp(np.log(trend["ROR"]) - 1.96 * np.sqrt(1/a + 1/b + 1/c + 1/d))
        trend["ROR_95CI_high"] = np.exp(np.log(trend["ROR"]) + 1.96 * np.sqrt(1/a + 1/b + 1/c + 1/d))

    return trend


def plot_enhanced_forest_plot(sig: pd.DataFrame, out_dir: Path) -> None:
    """Create enhanced forest plot with better styling and additional information."""
    if sig.empty:
        return

    # Sort by ROR for better visualization
    plot_df = sig.sort_values("ROR", ascending=False)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})

    # Main forest plot
    colors = ["#ff0000" if s else "#0000ff" for s in plot_df["signal_ROR"]]
    ax1.errorbar(
        plot_df["ROR"],
        plot_df["drug_flag"],
        xerr=[plot_df["ROR"] - plot_df["ROR_95CI_low"], plot_df["ROR_95CI_high"] - plot_df["ROR"]],
        fmt="o",
        capsize=5,
        capthick=2,
        color=colors,
        ecolor="gray",
        markersize=10,
    )
    ax1.axvline(1, linestyle="--", color="black", alpha=0.5, linewidth=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("ROR (log scale)", fontsize=12)
    ax1.set_ylabel("SGLT2 Inhibitor", fontsize=12)
    ax1.set_title("ROR Forest Plot with Signal Highlighting", fontsize=14, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Add ROR values as text
    for i, row in plot_df.iterrows():
        ax1.text(row["ROR"], i, f" {row['ROR']:.2f}", va='center', fontsize=9)

    # PRR subplot
    prr_colors = ["#ff0000" if s else "#0000ff" for s in plot_df["signal_PRR"]]
    ax2.errorbar(
        plot_df["PRR"],
        plot_df["drug_flag"],
        xerr=[plot_df["PRR"] - plot_df["PRR_95CI_low"], plot_df["PRR_95CI_high"] - plot_df["PRR"]],
        fmt="o",
        capsize=5,
        capthick=2,
        color=prr_colors,
        ecolor="gray",
        markersize=10,
    )
    ax2.axvline(1, linestyle="--", color="black", alpha=0.5, linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("PRR (log scale)", fontsize=12)
    ax2.set_title("PRR Forest Plot", fontsize=12)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "ror_prr_forest_plot.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run ROR/PRR signal detection and annual trend analysis.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--run-annual-trend", action="store_true", help="Run annual signal trend analysis")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    try:
        all_cases = pd.read_parquet(processed / "all_cases.parquet")
        model_dataset = pd.read_parquet(processed / "model_dataset.parquet")

        print(f"Loaded all_cases: {len(all_cases)} records")
        print(f"Loaded model_dataset: {len(model_dataset)} records")

        drug_flags = [c for c in all_cases.columns if c.startswith("has_") and c not in {"has_study_drug_any"}]
        print(f"Found {len(drug_flags)} drug flags to analyze")

        # Run signal detection
        sig = signal_for_drug_flags(all_cases, drug_flags, label_col="label_target_event")
        sig.to_csv(out / "tables" / "signal_detection.csv", index=False)
        print(f"\nSignal detection results saved to {out}/tables/signal_detection.csv")
        print("\nTop 5 drugs by ROR:")
        print(sig.head().to_string(index=False))

        # Create enhanced forest plot
        plot_enhanced_forest_plot(sig, out / "figures")
        print(f"Forest plot saved to {out}/figures/ror_prr_forest_plot.png")

        # Run annual trend analysis if requested
        if args.run_annual_trend:
            print("\nRunning annual signal trend analysis...")
            annual_results = []

            for flag in drug_flags:
                drug_name = flag.replace("has_", "")
                trend = annual_signal_trend(all_cases, flag, label_col="label_target_event")

                if not trend.empty and len(trend) > 1:
                    trend["drug_flag"] = drug_name
                    annual_results.append(trend)

            if annual_results:
                annual_df = pd.concat(annual_results)
                annual_df.to_csv(out / "tables" / "signal_by_year.csv", index=False)
                print(f"Annual signal trends saved to {out}/tables/signal_by_year.csv")

                # Plot annual trends for top 3 drugs
                top_drugs = sig.head(3)["drug_flag"].tolist()
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                for i, drug in enumerate(top_drugs):
                    drug_trend = annual_df[annual_df["drug_flag"] == drug]
                    if not drug_trend.empty:
                        axes[i].plot(drug_trend["report_year"], drug_trend["ROR"], marker='o', label='ROR')
                        axes[i].axhline(1, linestyle='--', color='gray', alpha=0.5)
                        axes[i].set_title(f"{drug} - Annual ROR Trend")
                        axes[i].set_xlabel("Report Year")
                        axes[i].set_ylabel("ROR")
                        axes[i].set_yscale("log")
                        axes[i].grid(True, alpha=0.3)
                        axes[i].legend()

                plt.tight_layout()
                fig.savefig(out / "figures" / "annual_signal_trends.png", dpi=300)
                plt.close(fig)
                print(f"Annual trend plots saved to {out}/figures/annual_signal_trends.png")
            else:
                print("No annual trends could be calculated (insufficient data)")

        print("\n" + "=" * 60)
        print("Signal detection completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Signal detection failed. Please check the error message and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
