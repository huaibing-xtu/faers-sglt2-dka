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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import tree models, if not available use RandomForest
try:
    from lightgbm import LGBMClassifier
    USE_LGBM = True
    BEST_MODEL_NAME = "LightGBM"
except ImportError:
    USE_LGBM = False
    try:
        from catboost import CatBoostClassifier
        USE_CATBOOST = True
        BEST_MODEL_NAME = "CatBoost"
    except ImportError:
        USE_CATBOOST = False
        BEST_MODEL_NAME = "RandomForest"

EXCLUDE_PREFIXES = {"outcome_", "any_serious", "label_", "reaction_count"}
EXCLUDE_EXACT = {"primaryid", "caseid", "quarter"}
EXCLUDE_PATTERNS = ["target_event", "dka", "ketoacidosis", "death", "hospitalization", "serious", "life_threatening", "outcome"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) for modeling."""
    numeric = [c for c in ["age_num", "report_year", "drug_count", "indication_count"] if c in df.columns]
    categorical = [
        c for c in [
            "sex", "age_cod", "age_grp", "occp_cod", "reporter_country", "occr_country",
            "study_drug_main", "study_drug_role", "study_drug_route", "reporter_type",
        ] if c in df.columns
    ]
    binary_prefixes = ["concomitant_", "ind_"]
    binary = [
        c for c in df.columns
        if any(c.startswith(p) for p in binary_prefixes)
        and not any(c.startswith(ep) for ep in EXCLUDE_PREFIXES)
        and c not in EXCLUDE_EXACT
        and not any(pattern in c for pattern in EXCLUDE_PATTERNS)
    ]
    # Final safety check
    all_features = numeric + categorical + binary
    safe_features = [
        c for c in all_features
        if not any(c.startswith(ep) for ep in EXCLUDE_PREFIXES)
        and c not in EXCLUDE_EXACT
        and not any(pattern in c for pattern in EXCLUDE_PATTERNS)
    ]
    numeric = [c for c in numeric if c in safe_features]
    categorical = [c for c in categorical if c in safe_features]
    binary = [c for c in binary if c in safe_features]
    return numeric, categorical + binary


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build preprocessing pipeline for the model."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def temporal_split(df: pd.DataFrame, label_col: str,
                  train_year_end: int = 2021,
                  valid_year: int = 2022,
                  test_year_start: int = 2023) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets based on report_year."""
    print(f"Temporal split:")
    print(f"  Train set: report_year <= {train_year_end}")
    print(f"  Validation set: report_year == {valid_year}")
    print(f"  Test set: report_year >= {test_year_start}")

    # Convert report_year to numeric, handle errors
    df["report_year"] = pd.to_numeric(df["report_year"], errors="coerce")

    # Split data
    train_df = df[df["report_year"] <= train_year_end].copy()
    valid_df = df[df["report_year"] == valid_year].copy()
    test_df = df[df["report_year"] >= test_year_start].copy()

    print(f"  Train set size: {len(train_df)}")
    print(f"  Validation set size: {len(valid_df)}")
    print(f"  Test set size: {len(test_df)}")

    # Check validation set size
    if len(valid_df) < 10:
        print("Warning: Validation set too small (<10 samples). Merging with training set.")
        train_df = pd.concat([train_df, valid_df], ignore_index=True)
        valid_df = pd.DataFrame()  # Empty validation set
        print(f"  Updated train set size: {len(train_df)}")

    return train_df, valid_df, test_df


def train_best_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
                    label_col: str, random_state: int = 42) -> Pipeline:
    """Train the best available model"""
    # Get feature columns
    numeric_features, categorical_features = get_feature_columns(train_df)
    feature_cols = numeric_features + categorical_features

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Filter data to include only selected features and SGLT2 reports
    def filter_data(df):
        df = df[df["has_study_drug_any"].fillna(0).astype(int).eq(1)].copy()
        df[label_col] = df[label_col].fillna(0).astype(int)
        if df[label_col].nunique() < 2:
            raise RuntimeError("The dataset has only one class. Check target event terms.")
        return df[feature_cols], df[label_col]

    # Prepare data
    X_train, y_train = filter_data(train_df)
    X_valid, y_valid = filter_data(valid_df) if not valid_df.empty else (None, None)
    X_test, y_test = filter_data(test_df)

    print(f"  Feature count: {len(feature_cols)}")
    print(f"  Training positive samples: {y_train.sum()}/{len(y_train)}")

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
    elif USE_CATBOOST:
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=random_state
        )
    else:
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )

    # Create pipeline
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

    # Train model
    print(f"Training {BEST_MODEL_NAME}...")
    pipe.fit(X_train, y_train)

    # Validate if validation set is available
    if X_valid is not None:
        valid_pred = pipe.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_pred)
        print(f"Validation ROC-AUC: {valid_auc:.4f}")

    return pipe


def evaluate_model(pipe: Pipeline, test_df: pd.DataFrame, label_col: str) -> dict:
    """Evaluate model on test set"""
    # Filter test data
    test_df = test_df[test_df["has_study_drug_any"].fillna(0).astype(int).eq(1)].copy()
    test_df[label_col] = test_df[label_col].fillna(0).astype(int)

    numeric_features, categorical_features = get_feature_columns(test_df)
    feature_cols = numeric_features + categorical_features

    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    # Make predictions
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "PR_AUC": average_precision_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    })

    # Create predictions DataFrame
    predictions = pd.DataFrame({
        "y_true": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred
    })

    return metrics, predictions


def plot_curves(metrics: dict, test_df: pd.DataFrame, label_col: str, pipe: Pipeline, out_dir: Path) -> None:
    """Plot ROC and PR curves"""
    numeric_features, categorical_features = get_feature_columns(test_df)
    feature_cols = numeric_features + categorical_features

    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    # Get predictions
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.interp(fpr, [0, 1-metrics["TN"]/(metrics["TN"]+metrics["FP"]), 1], [0, 1-metrics["FN"]/(metrics["FN"]+metrics["TP"]), 1])
    roc_auc = metrics["ROC_AUC"]

    ax1.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # PR curve
    pr_auc = metrics["PR_AUC"]
    ax2.plot(fpr, tpr, label=f'PR curve (AUC = {pr_auc:.2f})', color='blue', lw=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "temporal_roc_pr_curves.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run temporal validation for FAERS SGLT2-DKA model.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory containing processed data")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for results")
    parser.add_argument("--train-year-end", type=int, default=2021, help="Year to split train set (default: 2021)")
    parser.add_argument("--valid-year", type=int, default=2022, help="Year for validation set (default: 2022)")
    parser.add_argument("--test-year-start", type=int, default=2023, help="Year to start test set (default: 2023)")
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
    print("FAERS SGLT2-DKA Temporal Validation")
    print("=" * 60)
    print(f"Processed directory: {processed_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Train year end: {args.train_year_end}")
    print(f"Validation year: {args.valid_year}")
    print(f"Test year start: {args.test_year_start}")
    print(f"Random state: {args.random_state}")
    print("-" * 60)

    try:
        # Load data
        model_dataset = pd.read_parquet(processed_dir / "model_dataset.parquet")
        print(f"Loaded model_dataset: {len(model_dataset)} records")

        # Temporal split
        train_df, valid_df, test_df = temporal_split(
            model_dataset, "label_target_event",
            args.train_year_end, args.valid_year, args.test_year_start
        )

        # Train model
        pipe = train_best_model(train_df, valid_df, test_df, "label_target_event", args.random_state)

        # Evaluate on test set
        metrics, predictions = evaluate_model(pipe, test_df, "label_target_event")

        # Save results
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(tables_dir / "temporal_validation_metrics.csv", index=False)
        print(f"\nValidation metrics saved to {tables_dir}/temporal_validation_metrics.csv")

        predictions.to_csv(tables_dir / "temporal_test_predictions.csv", index=False)
        print(f"Test predictions saved to {tables_dir}/temporal_test_predictions.csv")

        # Plot curves
        plot_curves(metrics, test_df, "label_target_event", pipe, figures_dir)
        print(f"ROC/PR curves saved to {figures_dir}/temporal_roc_pr_curves.png")

        # Print results
        print("\nTemporal Validation Results:")
        print("=" * 60)
        print("Test Set Performance:")
        for key, value in metrics.items():
            if key in ["TN", "FP", "FN", "TP"]:
                print(f"  {key}: {int(value)}")
            else:
                print(f"  {key}: {value:.4f}")

        print(f"\nModel: {BEST_MODEL_NAME}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Temporal validation failed. Please check the error message and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()