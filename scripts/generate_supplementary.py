"""
生成Supplementary Materials:
1. SHAP Dependence Plots (前10个重要特征)
2. Calibration Curve
3. 完整特征列表及定义

输出目录: outputs/paper_results/supplementary/
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
import seaborn as sns
import joblib

ROOT = Path(r'E:\FAERS_DKA')
DATA = ROOT / 'data' / 'processed'
OUT  = ROOT / 'outputs' / 'paper_results' / 'supplementary'
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  SUPPLEMENTARY MATERIALS GENERATION")
print("=" * 70)

# ──────── Load Data ────────
print("\n1. Loading data...")
df = pd.read_parquet(DATA / "model_dataset.parquet")
sglt2 = df[df["has_study_drug_any"].eq(1)].copy()
sglt2["label_target_event"] = sglt2["label_target_event"].fillna(0).astype(int)

# Exclude outcome variables
EXCLUDE_PREFIX = {'any_serious', 'outcome_'}
EXCLUDE_EXACT  = {'primaryid', 'caseid', 'caseversion', 'quarter',
                  'label_target_event', 'has_study_drug_any',
                  'fda_dt', 'fda_dt_parsed', 'event_dt', 'init_fda_dt', 'init_fda_dt_parsed'}

def s(x):
    return pd.to_numeric(x, errors='coerce').fillna(0)

# Feature engineering (same as paper_pipeline.py)
raw_cols = [c for c in sglt2.columns
            if not any(c.startswith(p) for p in EXCLUDE_PREFIX) and c not in EXCLUDE_EXACT]
fe = sglt2[raw_cols].copy()

# Clean numeric
for c in ['age_num', 'drug_count', 'indication_count', 'report_year']:
    if c in fe.columns: fe[c] = s(fe[c])

# Binary
for c in fe.columns:
    if any(c.startswith(p) for p in ['has_', 'concomitant_', 'ind_']):
        fe[c] = s(fe[c]).astype(int)

# Label-encode strings
for c in fe.select_dtypes(include=['object']).columns:
    fe[c] = pd.Categorical(fe[c]).codes.astype(float)

# ── Engineered features ──
print("2. Creating engineered features...")
sglt2_drugs  = [c for c in fe.columns if c.startswith('has_')]
concomitants = [c for c in fe.columns if c.startswith('concomitant_')]
indications  = [c for c in fe.columns if c.startswith('ind_')]

# Drug risk profiles
risk_w = {'has_canagliflozin':1.5,'has_empagliflozin':1.0,'has_dapagliflozin':0.8,
          'has_ertugliflozin':0.7,'has_ipragliflozin':0.5,'has_luseogliflozin':0.5,
          'has_sotagliflozin':0.6,'has_tofogliflozin':0.5}
fe['sglt2_risk_weighted'] = sum(fe.get(c,0)*w for c,w in risk_w.items())
fe['sglt2_count'] = fe[sglt2_drugs].sum(axis=1) if sglt2_drugs else 0

# Age features
if 'age_num' in fe.columns:
    a = s(fe['age_num'])
    fe['age_squared'] = a**2; fe['age_log'] = np.log1p(a); fe['age_sqrt'] = np.sqrt(a.clip(0))
    for lo, hi, nm in [(0,18,'lt18'),(18,45,'18_44'),(45,65,'45_64'),(65,75,'65_74'),(75,200,'75plus')]:
        fe[f'age_bin_{nm}'] = ((a>=lo)&(a<hi)).astype(int)

# Drug interactions
fe['concomitant_total'] = fe[concomitants].sum(axis=1) if concomitants else 0
if 'concomitant_insulin' in fe.columns and 'concomitant_diuretic' in fe.columns:
    fe['insulin_diuretic'] = fe['concomitant_insulin'] * fe['concomitant_diuretic']
if 'concomitant_insulin' in fe.columns:
    for d in sglt2_drugs[:4]:
        fe[f'{d}_insulin'] = fe[d].astype(int) * fe['concomitant_insulin']
if 'drug_count' in fe.columns:
    dc = s(fe['drug_count'])
    fe['drug_burden_2'] = dc**2; fe['drug_count_log'] = np.log1p(dc)
if 'drug_count' in fe.columns and 'age_num' in fe.columns:
    fe['age_drug_product'] = s(fe['age_num']) * s(fe['drug_count'])

# Indication features
fe['total_indications'] = fe[indications].sum(axis=1) if indications else 0
if 'ind_diabetes' in fe.columns:
    if 'ind_chronic_kidney_disease' in fe.columns:
        fe['dm_ckd'] = fe['ind_diabetes'] * fe['ind_chronic_kidney_disease']
    if 'ind_heart_failure' in fe.columns:
        fe['dm_hf'] = fe['ind_diabetes'] * fe['ind_heart_failure']
chronic_cols = [c for c in indications if any(t in c.lower() for t in
    ['renal','kidney','heart','cardiac','liver','hepatic','hypertens','hyperlipid','obesity','neuropath','failure'])]
if chronic_cols:
    fe['comorbidity_score'] = fe[chronic_cols].sum(axis=1)
if 'drug_count' in fe.columns and 'indication_count' in fe.columns:
    fe['drugs_per_indication'] = s(fe['drug_count'])/(s(fe['indication_count']).clip(1))

# Reporter/temporal features
if 'report_year' in fe.columns:
    ry = s(fe['report_year'])
    fe['years_since_2013'] = (ry-2013).clip(0)
    fe['post_fda_warning'] = (ry>=2015).astype(int)
    fe['covid_era'] = (ry>=2020).astype(int)
if 'reporter_type' in fe.columns:
    fe['is_healthcare'] = fe['reporter_type'].isin([0,1,2]).astype(int)

# Clean up
fe = fe.select_dtypes(include=['int64','float64','int32','float32']).fillna(0)
fe = fe.replace([np.inf,-np.inf], 0)
const_cols = [c for c in fe.columns if fe[c].nunique()<=1]
if const_cols: fe = fe.drop(columns=const_cols)

X = fe; y = sglt2["label_target_event"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale + Select
scaler = RobustScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
K = min(80, X_train_s.shape[1])
selector = SelectKBest(f_classif, k=K)
X_tr = pd.DataFrame(selector.fit_transform(X_train_s, y_train),
                    columns=X_train_s.columns[selector.get_support()])
X_te = pd.DataFrame(selector.transform(X_test_s),
                    columns=X_train_s.columns[selector.get_support()])

print(f"   Features: {X.shape[1]} raw → {K} selected")

# ──────── Train Model ────────
print("\n3. Training LightGBM model...")
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7,
                     class_weight='balanced', min_child_samples=50,
                     subsample=0.85, colsample_bytree=0.85,
                     reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
lgb.fit(X_tr, y_train)
y_prob = lgb.predict_proba(X_te)[:, 1]

print(f"   Model trained. Test PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

# ──────── 1. SHAP Dependence Plots ────────
print("\n4. Generating SHAP Dependence Plots...")
try:
    import shap
    X_explain = X_te.sample(min(2000, len(X_te)), random_state=42)
    explainer = shap.TreeExplainer(lgb)
    shap_values = explainer.shap_values(X_explain)

    # Get top 10 features by mean absolute SHAP value
    shap_importance = pd.DataFrame({
        'feature': X_explain.columns,
        'shap_importance': np.abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)

    top_10_features = shap_importance.head(10)['feature'].tolist()

    # ── Part 1: Top 1-6 features (2×3 grid) ──
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 9))
    axes1 = axes1.flatten()
    for idx, feature in enumerate(top_10_features[:6]):
        ax = axes1[idx]
        shap.dependence_plot(
            feature, shap_values, X_explain,
            ax=ax, show=False
        )
        ax.set_title(feature, fontsize=24)

    plt.tight_layout()
    plt.savefig(OUT / 'shap_dependence_top10_part1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUT / 'shap_dependence_top10_part1.png'}")

    # ── Part 2: Top 7-10 features (2×2 grid) ──
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    axes2 = axes2.flatten()
    for idx, feature in enumerate(top_10_features[6:]):
        ax = axes2[idx]
        shap.dependence_plot(
            feature, shap_values, X_explain,
            ax=ax, show=False
        )
        ax.set_title(feature, fontsize=24)

    plt.tight_layout()
    plt.savefig(OUT / 'shap_dependence_top10_part2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {OUT / 'shap_dependence_top10_part2.png'}")

    # Save SHAP importance table
    shap_importance.to_csv(OUT / 'shap_importance.csv', index=False)
    print(f"   Saved: {OUT / 'shap_importance.csv'}")

except Exception as e:
    print(f"   SHAP dependence plots failed: {e}")

# ──────── 2. Calibration Curve ────────
print("\n5. Generating Calibration Curve...")
from sklearn.calibration import calibration_curve

# Calculate calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='LightGBM', color='#E53935', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', alpha=0.5)
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives', fontsize=12)
ax.set_title('Calibration Curve - LightGBM Model', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Add Brier score
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_test, y_prob)
ax.text(0.05, 0.95, f'Brier Score: {brier:.4f}', transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUT / 'calibration_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {OUT / 'calibration_curve.png'}")
print(f"   Brier Score: {brier:.4f}")

# ──────── 3. Feature List ────────
print("\n6. Generating Feature List...")

# Create feature definitions
feature_definitions = {
    'age_num': 'Patient age in years (continuous)',
    'drug_count': 'Number of drugs reported in the case',
    'indication_count': 'Number of indications reported',
    'report_year': 'Year of the FDA report',
    'sex': 'Patient sex (categorical)',
    'age_cod': 'Age unit code (YR/MO/WK/DY/HR)',
    'age_grp': 'Age group category',
    'occp_cod': 'Reporter occupation code',
    'reporter_country': 'Country of the reporter',
    'occr_country': 'Country where the event occurred',
    'reporter_type': 'Type of reporter (healthcare/consumer/manufacturer)',
    'study_drug_main': 'Main SGLT2 inhibitor name',
    'study_drug_role': 'Role of the study drug (primary/concomitant)',
    'study_drug_route': 'Route of administration',
    'concomitant_insulin': 'Concomitant insulin use (binary)',
    'concomitant_metformin': 'Concomitant metformin use (binary)',
    'concomitant_diuretic': 'Concomitant diuretic use (binary)',
    'concomitant_nsaid': 'Concomitant NSAID use (binary)',
    'concomitant_steroid': 'Concomitant steroid use (binary)',
    'ind_diabetes': 'Diabetes indication (binary)',
    'ind_chronic_kidney_disease': 'Chronic kidney disease indication (binary)',
    'ind_heart_failure': 'Heart failure indication (binary)',
    'has_canagliflozin': 'Canagliflozin reported (binary)',
    'has_dapagliflozin': 'Dapagliflozin reported (binary)',
    'has_empagliflozin': 'Empagliflozin reported (binary)',
    'has_ertugliflozin': 'Ertugliflozin reported (binary)',
    'has_ipragliflozin': 'Ipragliflozin reported (binary)',
    'has_luseogliflozin': 'Luseogliflozin reported (binary)',
    'has_sotagliflozin': 'Sotagliflozin reported (binary)',
    'has_tofogliflozin': 'Tofogliflozin reported (binary)',
    'sglt2_risk_weighted': 'SGLT2 risk-weighted score (weighted by DKA association strength)',
    'sglt2_count': 'Number of SGLT2 inhibitors reported',
    'age_squared': 'Age² (non-linear age effect)',
    'age_log': 'Log-transformed age',
    'age_sqrt': 'Square root of age',
    'age_bin_lt18': 'Age <18 years (binary)',
    'age_bin_18_44': 'Age 18-44 years (binary)',
    'age_bin_45_64': 'Age 45-64 years (binary)',
    'age_bin_65_74': 'Age 65-74 years (binary)',
    'age_bin_75plus': 'Age ≥75 years (binary)',
    'concomitant_total': 'Total number of concomitant medications',
    'insulin_diuretic': 'Interaction: insulin × diuretic use',
    'drug_burden_2': 'Drug count² (non-linear drug burden)',
    'drug_count_log': 'Log-transformed drug count',
    'age_drug_product': 'Interaction: age × drug count',
    'total_indications': 'Total number of indications',
    'dm_ckd': 'Interaction: diabetes × chronic kidney disease',
    'dm_hf': 'Interaction: diabetes × heart failure',
    'comorbidity_score': 'Sum of chronic condition indicators',
    'drugs_per_indication': 'Ratio of drug count to indication count',
    'years_since_2013': 'Years since 2013 (temporal trend)',
    'post_fda_warning': 'Post-FDA warning period (≥2015, binary)',
    'covid_era': 'COVID-19 era (≥2020, binary)',
    'is_healthcare': 'Healthcare professional reporter (binary)',
}

# Create feature list DataFrame
feature_list = pd.DataFrame([
    {
        'Feature': col,
        'Type': 'Numeric' if col in ['age_num', 'drug_count', 'indication_count', 'report_year'] else
                'Binary' if col.startswith(('has_', 'concomitant_', 'ind_', 'age_bin_', 'dm_', 'post_', 'covid_', 'is_')) else
                'Engineered' if col in ['sglt2_risk_weighted', 'sglt2_count', 'age_squared', 'age_log', 'age_sqrt',
                                        'drug_burden_2', 'drug_count_log', 'age_drug_product', 'total_indications',
                                        'comorbidity_score', 'drugs_per_indication', 'years_since_2013',
                                        'concomitant_total', 'insulin_diuretic'] else
                'Categorical',
        'Description': feature_definitions.get(col, 'Engineered feature'),
        'Selected': col in X_tr.columns,
        'SHAP_Importance': shap_importance.loc[shap_importance['feature'] == col, 'shap_importance'].values[0]
                          if col in shap_importance['feature'].values else 0
    }
    for col in X.columns
])

# Sort by SHAP importance
feature_list = feature_list.sort_values('SHAP_Importance', ascending=False)

# Save feature list
feature_list.to_csv(OUT / 'feature_list.csv', index=False)
print(f"   Saved: {OUT / 'feature_list.csv'}")

# Create feature summary
summary = {
    'total_features': len(X.columns),
    'selected_features': len(X_tr.columns),
    'numeric_features': len([c for c in X.columns if feature_list.loc[feature_list['Feature'] == c, 'Type'].values[0] == 'Numeric']),
    'binary_features': len([c for c in X.columns if feature_list.loc[feature_list['Feature'] == c, 'Type'].values[0] == 'Binary']),
    'categorical_features': len([c for c in X.columns if feature_list.loc[feature_list['Feature'] == c, 'Type'].values[0] == 'Categorical']),
    'engineered_features': len([c for c in X.columns if feature_list.loc[feature_list['Feature'] == c, 'Type'].values[0] == 'Engineered']),
}

# Save summary
import json
with open(OUT / 'feature_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   Saved: {OUT / 'feature_summary.json'}")

# Print summary
print(f"\n   Feature Summary:")
print(f"   - Total features: {summary['total_features']}")
print(f"   - Selected features: {summary['selected_features']}")
print(f"   - Numeric: {summary['numeric_features']}")
print(f"   - Binary: {summary['binary_features']}")
print(f"   - Categorical: {summary['categorical_features']}")
print(f"   - Engineered: {summary['engineered_features']}")

print(f"\n{'='*70}")
print(f"  SUPPLEMENTARY MATERIALS COMPLETE")
print(f"{'='*70}")
print(f"  Output directory: {OUT}/")
print(f"  ├── shap_dependence_top10.png")
print(f"  ├── shap_importance.csv")
print(f"  ├── calibration_curve.png")
print(f"  ├── feature_list.csv")
print(f"  └── feature_summary.json")
print(f"Done.")
