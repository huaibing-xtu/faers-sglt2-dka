from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_shap(model_dataset: pd.DataFrame, out_dir: str | Path, max_samples: int = 3000) -> None:
    """Run SHAP on the saved best model. Works best for tree-based models."""
    import shap

    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    bundle_path = out_dir / "best_model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Best model not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    pipe = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]
    label_col = bundle.get("label_col", "label_target_event")

    # Check for test_predictions.csv to select high-confidence positive samples
    test_pred_path = out_dir / "tables" / "test_predictions.csv"
    if test_pred_path.exists():
        test_predictions = pd.read_csv(test_pred_path)
        # Find the test sample with highest positive prediction
        high_conf_pos_idx = test_predictions[test_predictions["model"] == bundle["model_name"]].nlargest(1, "y_prob")
        if not high_conf_pos_idx.empty:
            print(f"Using test sample with highest positive prediction for waterfall plot")
            sample_idx = int(high_conf_pos_idx.iloc[0].name)
            if sample_idx < len(model_dataset):
                sample = model_dataset.iloc[[sample_idx]]
            else:
                sample = model_dataset.sample(n=1, random_state=42)
        else:
            sample = model_dataset.sample(n=1, random_state=42)
    else:
        # Random sample if no test predictions available
        sample = model_dataset.sample(n=1, random_state=42)

    df = model_dataset[model_dataset["has_study_drug_any"].fillna(0).astype(int).eq(1)].copy()
    if len(df) > max_samples:
        # Stratified-like sample: keep all positives if possible plus random negatives.
        pos = df[df[label_col].eq(1)]
        neg = df[df[label_col].eq(0)]
        n_pos = min(len(pos), max_samples // 2)
        n_neg = max_samples - n_pos
        sample = pd.concat([
            pos.sample(n=n_pos, random_state=42) if len(pos) > n_pos else pos,
            neg.sample(n=min(len(neg), n_neg), random_state=42) if len(neg) > 0 else neg,
        ], ignore_index=True)
    else:
        sample = df.copy()

    X_raw = sample[feature_cols]
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    X = preprocess.transform(X_raw)
    feature_names = preprocess.get_feature_names_out()

    # shap can be slow for non-tree models. Try tree explainer first; fall back to generic explainer.
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            values = shap_values[1]
        else:
            values = shap_values
            if values.ndim == 3:
                values = values[:, :, 1]
    except Exception as exc:
        print(f"[warn] TreeExplainer failed ({exc}); using model-agnostic Explainer on a small sample.")
        f = lambda data: model.predict_proba(data)[:, 1]
        background = shap.sample(X, min(200, X.shape[0]), random_state=42)
        explainer = shap.Explainer(f, background)
        shap_exp = explainer(X[: min(500, X.shape[0])])
        values = shap_exp.values
        X = X[: min(500, X.shape[0])]

    # Feature importance table
    mean_abs = np.abs(values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out_dir / "tables" / "shap_feature_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(values, X, feature_names=feature_names, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(values, X, feature_names=feature_names, plot_type="bar", show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Waterfall for first positive sample if available.
    pos_idx = np.where(sample[label_col].values[: X.shape[0]] == 1)[0]
    if len(pos_idx) > 0:
        i = int(pos_idx[0])
    else:
        i = 0
    try:
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        exp = shap.Explanation(values=values[i], base_values=base_value, data=X[i], feature_names=feature_names)
        shap.plots.waterfall(exp, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "shap_waterfall_example.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[warn] Failed to create waterfall plot: {exc}")
