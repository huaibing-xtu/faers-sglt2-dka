from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibrationDisplay


def optional_models(y_train) -> dict[str, Any]:
    """Return optional gradient boosting models if installed."""
    models: dict[str, Any] = {}
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
    except Exception as exc:
        print(f"[warn] XGBoost unavailable: {exc}")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    except Exception as exc:
        print(f"[warn] LightGBM unavailable: {exc}")

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=42,
        )
    except Exception as exc:
        print(f"[warn] CatBoost unavailable: {exc}")

    return models


# Columns that must NEVER be used as model features to prevent label leakage.
# - outcome_* / any_serious_*: post-event outcomes, not available at prediction time
# - label_*: the target itself
# - reaction_count: DKA reports naturally have more PTs → indirect label leakage
# - primaryid / caseid / quarter: identifiers, not predictive features
EXCLUDE_PREFIXES = {"outcome_", "any_serious", "label_", "reaction_count"}
EXCLUDE_EXACT = {"primaryid", "caseid", "quarter"}


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (numeric_features, categorical_features) for modeling.

    Features deliberately exclude:
    - Target-event text and reaction_count (label leakage)
    - Outcome fields (post-event, not available at prediction time)
    - Identifiers (primaryid, caseid, quarter)
    - Any column containing "target_event", "dka", "ketoacidosis", "death", "hospitalization", "serious", "life_threatening", "outcome"
    """
    # Define exclusion patterns
    EXCLUDE_PREFIXES = {"outcome_", "any_serious", "label_", "reaction_count"}
    EXCLUDE_EXACT = {"primaryid", "caseid", "quarter"}
    EXCLUDE_PATTERNS = ["target_event", "dka", "ketoacidosis", "death", "hospitalization", "serious", "life_threatening", "outcome"]

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

    # Final safety check: remove any column that matches exclusion patterns
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
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def train_models(model_dataset: pd.DataFrame, out_dir: str | Path, label_col: str = "label_target_event",
                 test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = model_dataset.copy()
    df = df[df["has_study_drug_any"].fillna(0).astype(int).eq(1)].copy()
    df[label_col] = df[label_col].fillna(0).astype(int)

    if df[label_col].nunique() < 2:
        raise RuntimeError("The model dataset has only one class. Check target event terms or drug matching.")

    numeric_features, categorical_features = get_feature_columns(df)
    feature_cols = numeric_features + categorical_features
    X = df[feature_cols].copy()
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    base_models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
    }
    base_models.update(optional_models(y_train))

    results = []
    cv_results_list = []
    fitted = {}
    test_predictions = []

    plt.figure()
    ax_roc = plt.gca()
    plt.figure()
    ax_pr = plt.gca()
    plt.figure()
    ax_cal = plt.gca()

    # --- 5-fold stratified cross-validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scoring = ["roc_auc", "average_precision", "f1"]

    for name, model in base_models.items():
        print(f"[train] {name}")
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        # Get predictions
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        row = {
            "model": name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "positive_train": int(y_train.sum()),
            "positive_test": int(y_test.sum()),
            "ROC_AUC": roc_auc_score(y_test, prob),
            "PR_AUC": average_precision_score(y_test, prob),
            "F1": f1_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Accuracy": accuracy,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        }

        # Cross-validation
        print(f"[cv] {name} — 5-fold stratified CV")
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=cv_scoring, return_train_score=False, n_jobs=-1,
        )
        cv_row = {"model": name}
        for metric in cv_scoring:
            scores = cv_res[f"test_{metric}"]
            cv_row[f"CV_mean_{metric}"] = scores.mean()
            cv_row[f"CV_std_{metric}"] = scores.std()
        cv_results_list.append(cv_row)
        row["CV_ROC_AUC_mean"] = cv_row["CV_mean_roc_auc"]
        row["CV_ROC_AUC_std"] = cv_row["CV_std_roc_auc"]
        row["CV_PR_AUC_mean"] = cv_row["CV_mean_average_precision"]
        row["CV_PR_AUC_std"] = cv_row["CV_std_average_precision"]

        results.append(row)
        fitted[name] = pipe

        # Store test predictions for later analysis
        test_pred_df = pd.DataFrame({
            'y_true': y_test,
            'y_prob': prob,
            'y_pred': pred,
            'model': name
        })
        test_predictions.append(test_pred_df)

        RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y_test, prob, name=name, ax=ax_pr)
        CalibrationDisplay.from_predictions(y_test, prob, name=name, ax=ax_cal, n_bins=10)

    ax_roc.set_title("ROC curves")
    ax_roc.figure.tight_layout()
    ax_roc.figure.savefig(out_dir / "figures" / "roc_curves.png", dpi=300)
    ax_pr.set_title("Precision-Recall curves")
    ax_pr.figure.tight_layout()
    ax_pr.figure.savefig(out_dir / "figures" / "pr_curves.png", dpi=300)
    ax_cal.set_title("Calibration curves")
    ax_cal.figure.tight_layout()
    ax_cal.figure.savefig(out_dir / "figures" / "calibration_curves.png", dpi=300)
    plt.close("all")

    # Save test predictions
    if test_predictions:
        test_pred_df = pd.concat(test_predictions)
        test_pred_df.to_csv(out_dir / "tables" / "test_predictions.csv", index=False)
        print(f"Test predictions saved to {out_dir}/tables/test_predictions.csv")

    metrics = pd.DataFrame(results).sort_values("PR_AUC", ascending=False)
    metrics.to_csv(out_dir / "tables" / "model_metrics.csv", index=False)

    cv_df = pd.DataFrame(cv_results_list)
    cv_df.to_csv(out_dir / "tables" / "cv_metrics.csv", index=False)

    best_name = metrics.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump({
        "model_name": best_name,
        "pipeline": best_model,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "label_col": label_col,
    }, out_dir / "best_model.joblib")
    print(f"[best] {best_name}")
    return metrics, fitted, test_pred_df if test_predictions else None
