from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM, if not available use RandomForest
try:
    from lightgbm import LGBMClassifier
    USE_LGBM = True
except ImportError:
    USE_LGBM = False


def define_feature_modules(model_dataset: pd.DataFrame) -> dict[str, list[str]]:
    """Define feature modules based on column names"""
    modules = {}

    # Patient features
    patient_features = ["age", "age_num", "sex"]
    patient_modules = [col for col in patient_features if col in model_dataset.columns]
    modules['patient'] = patient_modules
    if not patient_modules:
        print("Warning: No patient features found")

    # Report features
    report_features = ["report_year", "country", "reporter_type"]
    report_modules = []
    # Map column names
    for col in report_features:
        if col in model_dataset.columns:
            report_modules.append(col)
        elif col == "country":
            # Check for country columns
            country_cols = [c for c in model_dataset.columns if "country" in c]
            if country_cols:
                report_modules.extend(country_cols[:1])  # Take first available country column

    modules['report'] = report_modules
    if not report_modules:
        print("Warning: No report features found")

    # Drug features
    drug_features = ["study_drug_main", "study_drug_role", "study_drug_route"]
    drug_modules = [col for col in drug_features if col in model_dataset.columns]
    modules['drug'] = drug_modules
    if not drug_modules:
        print("Warning: No drug features found")

    # Concomitant features
    concomitant_modules = [col for col in model_dataset.columns if col.startswith("concomitant_")]
    if "drug_count" in model_dataset.columns:
        concomitant_modules.append("drug_count")
    modules['concomitant'] = concomitant_modules
    if not concomitant_modules:
        print("Warning: No concomitant features found")

    # Indication features
    indication_modules = [col for col in model_dataset.columns if col.startswith("ind_")]
    modules['indication'] = indication_modules
    if not indication_modules:
        print("Warning: No indication features found")

    return modules


def build_feature_combinations(modules: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build different feature combinations for ablation study"""
    combinations = {}

    # Single modules
    if 'patient' in modules:
        combinations['A1 = patient'] = modules['patient']

    # Two modules
    if 'patient' in modules and 'report' in modules:
        combinations['A2 = patient + report'] = modules['patient'] + modules['report']

    # Three modules
    if 'patient' in modules and 'report' in modules and 'drug' in modules:
        combinations['A3 = patient + report + drug'] = modules['patient'] + modules['report'] + modules['drug']

    # Four modules
    if all(m in modules for m in ['patient', 'report', 'drug', 'concomitant']):
        combinations['A4 = patient + report + drug + concomitant'] = (
            modules['patient'] + modules['report'] + modules['drug'] + modules['concomitant']
        )

    # All modules
    if all(m in modules for m in ['patient', 'report', 'drug', 'concomitant', 'indication']):
        combinations['A5 = patient + report + drug + concomitant + indication'] = (
            modules['patient'] + modules['report'] + modules['drug'] + modules['concomitant'] + modules['indication']
        )

    return combinations


def train_and_evaluate(model_dataset: pd.DataFrame, feature_cols: list[str], label_col: str,
                      test_size: float = 0.2, random_state: int = 42) -> dict[str, float]:
    """Train and evaluate a single model configuration"""
    # Filter data to include only selected features
    df = model_dataset.copy()
    df = df[df["has_study_drug_any"].fillna(0).astype(int).eq(1)].copy()
    df[label_col] = df[label_col].fillna(0).astype(int)

    # Check if label has both classes
    if df[label_col].nunique() < 2:
        raise RuntimeError(f"The dataset for this combination has only one class.")

    X = df[feature_cols].copy()
    y = df[label_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Build preprocessor
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_features)
        ],
        remainder="drop"
    )

    # Choose model
    if USE_LGBM:
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )

    # Create and train pipeline
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    # Make predictions
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'ROC_AUC': roc_auc_score(y_test, y_prob),
        'PR_AUC': average_precision_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'n_features': len(feature_cols)
    }

    return metrics


def run_ablation_study(model_dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """Run the complete ablation study"""
    # Define feature modules
    print("Defining feature modules...")
    modules = define_feature_modules(model_dataset)
    print("\nFeature modules found:")
    for module_name, cols in modules.items():
        print(f"  {module_name}: {len(cols)} features")

    # Build feature combinations
    combinations = build_feature_combinations(modules)
    print(f"\nFound {len(combinations)} feature combinations to test:")
    for comb_name, cols in combinations.items():
        print(f"  {comb_name}: {len(cols)} features")

    # Train and evaluate each combination
    results = []
    for comb_name, feature_cols in combinations.items():
        print(f"\nTraining {comb_name}...")
        try:
            metrics = train_and_evaluate(
                model_dataset, feature_cols, "label_target_event", test_size, random_state
            )
            result = {
                'combination': comb_name,
                'n_features': metrics['n_features'],
                'ROC_AUC': metrics['ROC_AUC'],
                'PR_AUC': metrics['PR_AUC'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1']
            }
            results.append(result)
            print(f"  PR-AUC: {metrics['PR_AUC']:.4f}, F1: {metrics['F1']:.4f}")
        except Exception as e:
            print(f"  Failed: {str(e)}")
            # Add failed result with NaN values
            result = {
                'combination': comb_name,
                'n_features': len(feature_cols),
                'ROC_AUC': np.nan,
                'PR_AUC': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1': np.nan
            }
            results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def plot_ablation_results(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot ablation results"""
    # Filter out combinations with NaN values
    valid_results = results_df.dropna()
    if valid_results.empty:
        print("No valid results to plot")
        return

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PR-AUC plot
    pr_auc_data = valid_results.sort_values('PR_AUC', ascending=False)
    bars1 = ax1.barh(pr_auc_data['combination'], pr_auc_data['PR_AUC'], color='skyblue')
    ax1.set_xlabel('PR-AUC')
    ax1.set_title('PR-AUC by Feature Combination')
    ax1.grid(axis='x', alpha=0.3)

    # Add PR-AUC values as text
    for i, v in enumerate(pr_auc_data['PR_AUC']):
        ax1.text(v, i, f' {v:.3f}', va='center')

    # F1 plot
    f1_data = valid_results.sort_values('F1', ascending=False)
    bars2 = ax2.barh(f1_data['combination'], f1_data['F1'], color='lightgreen')
    ax2.set_xlabel('F1 Score')
    ax2.set_title('F1 Score by Feature Combination')
    ax2.grid(axis='x', alpha=0.3)

    # Add F1 values as text
    for i, v in enumerate(f1_data['F1']):
        ax2.text(v, i, f' {v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(out_dir / 'ablation_results.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for FAERS SGLT2-DKA model.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory containing processed data")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for results")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    # Create output directories
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FAERS SGLT2-DKA Ablation Study")
    print("=" * 60)
    print(f"Processed directory: {processed_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Random state: {args.random_state}")
    print("-" * 60)

    try:
        # Load data
        model_dataset = pd.read_parquet(processed_dir / "model_dataset.parquet")
        print(f"Loaded model_dataset: {len(model_dataset)} records")

        # Run ablation study
        results_df = run_ablation_study(
            model_dataset, test_size=0.2, random_state=args.random_state
        )

        # Save results
        results_df.to_csv(tables_dir / "ablation_metrics.csv", index=False)
        print(f"\nAblation results saved to {tables_dir}/ablation_metrics.csv")

        # Plot results
        plot_ablation_results(results_df, figures_dir)
        print(f"Ablation plot saved to {figures_dir}/ablation_results.png")

        # Print summary
        if not results_df.empty:
            print("\nSummary Results:")
            print("=" * 60)
            print(results_df.round(4).to_string(index=False))

        print("\n" + "=" * 60)
        print("Ablation study completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Ablation study failed. Please check the error message and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()