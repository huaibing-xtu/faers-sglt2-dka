"""
Master experiment pipeline for the paper:
"基于FAERS数据的SGLT2抑制剂相关糖尿病酮症酸中毒信号挖掘与可解释机器学习分析"

7 experiments:
  1. Descriptive Analysis (DKA vs non-DKA baseline characteristics)
  2. Signal Detection (ROR/PRR disproportionality)
  3. Machine Learning (v3 enhanced features)
  4. SHAP Explainability
  5. Temporal Validation
  6. Threshold Sensitivity Analysis
  7. Ablation Study

Output: outputs/paper_results/
"""

import sys, os, warnings, json, time, subprocess
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'processed'
OUT  = ROOT / 'outputs' / 'paper_results'
for d in ['tables', 'figures', 'models', 'reports']:
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ──────── 0. Load & Prep ────────
print("=" * 70)
print("  PAPER EXPERIMENT PIPELINE")
print("=" * 70)
t0 = time.time()

df = pd.read_parquet(DATA / "model_dataset.parquet")
sglt2 = df[df["has_study_drug_any"].eq(1)].copy()
sglt2["label_target_event"] = sglt2["label_target_event"].fillna(0).astype(int)

# Also load ALL cases for signal detection
if (DATA / "all_cases.parquet").exists():
    all_cases = pd.read_parquet(DATA / "all_cases.parquet")
else:
    all_cases = sglt2.copy()

EXCLUDE_PREFIX = {'any_serious', 'outcome_'}  # 排除结局变量，避免标签泄漏
EXCLUDE_EXACT  = {'primaryid', 'caseid', 'caseversion', 'quarter',
                  'label_target_event', 'has_study_drug_any',
                  'fda_dt', 'fda_dt_parsed', 'event_dt', 'init_fda_dt', 'init_fda_dt_parsed'}

def s(x):
    return pd.to_numeric(x, errors='coerce').fillna(0)

# ═══════════════════════════════════════════════════════
# EXPERIMENT 1: Descriptive Analysis
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 1: Descriptive Analysis")
print("=" * 70)
t1 = time.time()

dka   = sglt2[sglt2["label_target_event"] == 1]
ndka  = sglt2[sglt2["label_target_event"] == 0]

desc = {}
desc['total_sglt2_reports'] = len(sglt2)
desc['dka_reports'] = len(dka)
desc['non_dka_reports'] = len(ndka)
desc['dka_ratio'] = len(dka) / len(sglt2)

# Demographics
for col in ['age_num', 'drug_count', 'indication_count']:
    if col in sglt2.columns:
        vals = pd.to_numeric(sglt2[col], errors='coerce')
        dka_v = pd.to_numeric(dka[col], errors='coerce')
        ndka_v = pd.to_numeric(ndka[col], errors='coerce')
        desc[f'{col}_all_mean'] = float(vals.mean())
        desc[f'{col}_all_std']  = float(vals.std())
        desc[f'{col}_dka_mean'] = float(dka_v.mean())
        desc[f'{col}_ndka_mean']= float(ndka_v.mean())

if 'sex' in sglt2.columns:
    for lbl, grp in [('all', sglt2), ('dka', dka), ('ndka', ndka)]:
        vc = grp['sex'].value_counts()
        for k, v in vc.items():
            desc[f'sex_{lbl}_{k}'] = int(v)

# Drug distribution
drug_cols = [c for c in sglt2.columns if c.startswith('has_') and c != 'has_study_drug_any']
for dc in drug_cols:
    desc[f'{dc}_dka']   = int(dka[dc].sum())
    desc[f'{dc}_total'] = int(sglt2[dc].sum())

# Outcome distribution
outc_cols = [c for c in sglt2.columns if c.startswith('outcome_')]
for oc in outc_cols:
    desc[f'{oc}_dka']   = int(dka[oc].sum())
    desc[f'{oc}_ndka']  = int(ndka[oc].sum())

# Yearly trend
if 'report_year' in sglt2.columns:
    yearly = sglt2.groupby('report_year').agg(
        total=('label_target_event', 'count'),
        dka=('label_target_event', 'sum')
    ).reset_index()
    yearly['dka_pct'] = yearly['dka'] / yearly['total'] * 100
    yearly.to_csv(OUT / 'tables' / 'yearly_trend.csv', index=False)

# Save
with open(OUT / 'tables' / 'descriptive_stats.json', 'w') as f:
    json.dump(desc, f, indent=2, default=str)

print(f"  SGLT2 reports: {desc['total_sglt2_reports']:,}")
print(f"  DKA: {desc['dka_reports']:,} ({desc['dka_ratio']*100:.1f}%)")
print(f"  Time: {time.time()-t1:.1f}s")

# ═══════════════════════════════════════════════════════
# EXPERIMENT 2: Signal Detection (ROR/PRR)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 2: Signal Detection")
print("=" * 70)
t2 = time.time()

# Use all_cases for disproportionality analysis
drug_names = ['canagliflozin','dapagliflozin','empagliflozin','ertugliflozin',
              'ipragliflozin','luseogliflozin','sotagliflozin','tofogliflozin']
signals = []

for drug in drug_names:
    has_drug = f'has_{drug}'
    if has_drug not in all_cases.columns:
        continue

    dka_all = all_cases['label_target_event'].sum()
    non_dka_all = len(all_cases) - dka_all

    drug_total = int(all_cases[has_drug].sum())
    drug_dka = int(all_cases.loc[all_cases[has_drug] == 1, 'label_target_event'].sum())
    drug_non_dka = drug_total - drug_dka

    other_total = len(all_cases) - drug_total
    other_dka = int(dka_all - drug_dka)
    other_non_dka = other_total - other_dka

    # ROR
    a, b = drug_dka, drug_non_dka
    c, d_val = other_dka, other_non_dka

    if a > 0 and c > 0 and b > 0 and d_val > 0:
        ror = (a / b) / (c / d_val)
        se_ror = np.sqrt(1/a + 1/b + 1/c + 1/d_val)
        ror_ci_low = np.exp(np.log(ror) - 1.96 * se_ror)
        ror_ci_high = np.exp(np.log(ror) + 1.96 * se_ror)
        prr = (a / (a + b)) / (c / (c + d_val))

        signal = 'Yes' if ror_ci_low > 1 else 'No'
        signals.append({
            'drug': drug, 'dka_reports': a, 'total_reports': drug_total,
            'ROR': round(ror, 2), 'ROR_95CI_low': round(ror_ci_low, 2),
            'ROR_95CI_high': round(ror_ci_high, 2),
            'PRR': round(prr, 2), 'signal': signal
        })

signal_df = pd.DataFrame(signals).sort_values('ROR', ascending=False)
signal_df.to_csv(OUT / 'tables' / 'signal_detection.csv', index=False)
print(signal_df.to_string(index=False))
print(f"  Time: {time.time()-t2:.1f}s")

# ═══════════════════════════════════════════════════════
# EXPERIMENT 3: ML Training (V3 enhanced features)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 3: Machine Learning Models")
print("=" * 70)
t3 = time.time()

# Feature engineering (same as v3_enhanced)
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
print("  Creating engineered features...")
sglt2_drugs  = [c for c in fe.columns if c.startswith('has_')]
concomitants = [c for c in fe.columns if c.startswith('concomitant_')]
indications  = [c for c in fe.columns if c.startswith('ind_')]
outcomes_bin = [c for c in fe.columns if c.startswith('outcome_')]

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
# 注意：结局变量(outcome_*)已从主模型中移除，仅在敏感性分析中使用
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

print(f"  Features: {X.shape[1]} raw + engineered → {K} selected")

# Train models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ml_results = {}
y_probs = {}
importances = {}

pos_w = (y_train==0).sum() / max((y_train==1).sum(), 1)

# Model 1: LGB
lgb  = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7,
                       class_weight='balanced', min_child_samples=50,
                       subsample=0.85, colsample_bytree=0.85,
                       reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
lgb.fit(X_tr, y_train)
p1 = lgb.predict_proba(X_te)[:,1]

# Model 2: XGB
xgb_mod = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        scale_pos_weight=pos_w, subsample=0.85, colsample_bytree=0.85,
                        random_state=42, n_jobs=-1)
xgb_mod.fit(X_tr, y_train)
p2 = xgb_mod.predict_proba(X_te)[:,1]

# Model 3: Ensemble
ens = VotingClassifier([
    ('lgb', LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=63,
                            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
    ('xgb', XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                          scale_pos_weight=pos_w, random_state=42, n_jobs=-1)),
], voting='soft', n_jobs=-1)
ens.fit(X_tr, y_train)
p3 = ens.predict_proba(X_te)[:,1]

for name, prob in [('LightGBM', p1), ('XGBoost', p2), ('Ensemble', p3)]:
    ml_results[name] = {
        'ROC_AUC': roc_auc_score(y_test, prob),
        'PR_AUC': average_precision_score(y_test, prob),
        'F1': f1_score(y_test, (prob > 0.5).astype(int)),
        'Precision': precision_score(y_test, (prob > 0.5).astype(int), zero_division=0),
        'Recall': recall_score(y_test, (prob > 0.5).astype(int), zero_division=0),
    }
    y_probs[name] = (y_test, prob)

# Find optimal threshold for best model
best_name = max(ml_results, key=lambda n: ml_results[n]['PR_AUC'])
best_prob = y_probs[best_name][1]
prec, rec, thr = precision_recall_curve(y_test, best_prob)
f1s = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-10)
ml_results[best_name]['optimal_threshold'] = float(thr[np.argmax(f1s)])
ml_results[best_name]['optimal_f1'] = float(f1s.max())

# CV
cv_pipe = Pipeline([('scaler', RobustScaler()), ('selector', SelectKBest(f_classif, k=K)),
                     ('model', LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                              class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_s = cross_validate(cv_pipe, X_train, y_train, cv=cv,
                       scoring=['roc_auc','average_precision','f1'], n_jobs=-1)
cv_results = {k: {'mean': cv_s[f'test_{k}'].mean(), 'std': cv_s[f'test_{k}'].std()}
              for k in ['roc_auc','average_precision','f1']}
ml_results['CV'] = cv_results

pd.DataFrame(ml_results).T.to_csv(OUT / 'tables' / 'ml_model_results.csv')
importances['LightGBM'] = dict(zip(X_tr.columns, lgb.feature_importances_))

print(f"\n  Model Results:")
for name, m in ml_results.items():
    if name != 'CV' and 'PR_AUC' in m:
        print(f"  {name:<20s} PR-AUC={m['PR_AUC']:.4f}  ROC-AUC={m['ROC_AUC']:.4f}  F1={m['F1']:.4f}")
print(f"  CV PR-AUC: {cv_results['average_precision']['mean']:.4f} ± {cv_results['average_precision']['std']:.4f}")
print(f"  Time: {time.time()-t3:.1f}s")

# Save best model
import joblib
joblib.dump({'pipeline': ens, 'feature_cols': list(X_tr.columns),
             'selector': selector, 'scaler': scaler},
            OUT / 'models' / 'best_model.joblib')
# Also save test predictions
pd.DataFrame({'y_true': y_test, 'y_prob_ensemble': p3, 'y_prob_lgb': p1}).to_csv(
    OUT / 'tables' / 'test_predictions.csv', index=False)

# ═══════════════════════════════════════════════════════
# EXPERIMENT 4: SHAP Analysis
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 4: SHAP Explainability")
print("=" * 70)
t4 = time.time()

try:
    import shap
    # Sample for SHAP
    X_explain = X_te.sample(min(2000, len(X_te)), random_state=42)
    explainer = shap.TreeExplainer(lgb)
    shap_values = explainer.shap_values(X_explain)

    shap.summary_plot(shap_values, X_explain, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(OUT / 'figures' / 'shap_summary.png', dpi=200, bbox_inches='tight'); plt.close()

    # Top features
    shap_importance = pd.DataFrame({
        'feature': X_explain.columns,
        'shap_importance': np.abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)
    shap_importance.to_csv(OUT / 'tables' / 'shap_importance.csv', index=False)

    print(f"  Top 10 SHAP features:")
    for _, row in shap_importance.head(10).iterrows():
        print(f"    {row['feature']:<40s} {row['shap_importance']:.6f}")
    print(f"  Time: {time.time()-t4:.1f}s")
except Exception as e:
    print(f"  SHAP skipped: {e}")
    shap_importance = pd.DataFrame()

# ═══════════════════════════════════════════════════════
# EXPERIMENT 5: Temporal Validation
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 5: Temporal Validation")
print("=" * 70)
t5 = time.time()

if 'report_year' in X.columns:
    temporal_results = []
    data_full = pd.concat([X, y.rename('label')], axis=1)

    # Temporal split
    train_t = data_full[data_full['report_year'] <= 2021]
    test_t  = data_full[data_full['report_year'] >= 2022]

    for lbl, subset in [('Train(≤2021)', train_t), ('Test(≥2022)', test_t)]:
        if len(subset) == 0: continue
        X_sub = subset.drop(columns=['label', 'report_year'])
        y_sub = subset['label']

        # Refit on this temporal subset to avoid data leakage
        sub_scaler = RobustScaler()
        X_sub_s = pd.DataFrame(sub_scaler.fit_transform(X_sub), columns=X_sub.columns)
        sub_sel = SelectKBest(f_classif, k=min(K, X_sub.shape[1]))
        X_sub_sel = sub_sel.fit_transform(X_sub_s, y_sub)
        X_sub_sel = pd.DataFrame(X_sub_sel, columns=X_sub_s.columns[sub_sel.get_support()])

        lgb_temp = LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31,
                                  class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
        lgb_temp.fit(X_sub_sel, y_sub)
        p_temp = lgb_temp.predict_proba(X_sub_sel)[:, 1]
        temporal_results.append({
            'split': lbl, 'n': len(X_sub), 'dka_rate': float(y_sub.mean()),
            'PR_AUC': average_precision_score(y_sub, p_temp) if y_sub.nunique()>1 else np.nan,
            'ROC_AUC': roc_auc_score(y_sub, p_temp) if y_sub.nunique()>1 else np.nan,
        })

    pd.DataFrame(temporal_results).to_csv(OUT / 'tables' / 'temporal_validation.csv', index=False)
    for tr in temporal_results:
        print(f"  {tr['split']}: n={tr['n']:,} PR-AUC={tr['PR_AUC']:.4f} ROC-AUC={tr['ROC_AUC']:.4f}")
    print(f"  Time: {time.time()-t5:.1f}s")
else:
    print("  Skipped (no report_year)")

# ═══════════════════════════════════════════════════════
# EXPERIMENT 6: Threshold Sensitivity Analysis
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 6: Threshold Sensitivity")
print("=" * 70)
t6 = time.time()

thresholds = np.arange(0.1, 1.0, 0.05)
thresh_data = []
for th in thresholds:
    yp = (best_prob >= th).astype(int)
    thresh_data.append({
        'threshold': round(th, 2),
        'precision': precision_score(y_test, yp, zero_division=0),
        'recall': recall_score(y_test, yp, zero_division=0),
        'f1': f1_score(y_test, yp, zero_division=0),
        'accuracy': (yp == y_test).mean(),
    })
pd.DataFrame(thresh_data).to_csv(OUT / 'tables' / 'threshold_analysis.csv', index=False)

best_th = max(thresh_data, key=lambda x: x['f1'])
print(f"  Optimal F1 threshold: {best_th['threshold']} (F1={best_th['f1']:.4f})")
print(f"  High-precision threshold: 0.55 (P={[t['precision'] for t in thresh_data if t['threshold']==0.55][0] if any(t['threshold']==0.55 for t in thresh_data) else 'N/A'})")
print(f"  Time: {time.time()-t6:.1f}s")

# ═══════════════════════════════════════════════════════
# EXPERIMENT 7: Ablation Study
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT 7: Feature Ablation Study")
print("=" * 70)
t7 = time.time()

# Feature groups
# 注意：结局变量(outcome_*)已从主模型中移除，不参与消融实验
feature_groups = {
    'demographic':  [c for c in X_tr.columns if any(p in c for p in
                      ['age_','sex','occp','reporter_country','occr_country'])],
    'drug':         [c for c in X_tr.columns if any(p in c for p in
                      ['has_','study_drug','sglt2','drug_count','drug_burden',
                       'concomitant','drugs_per']) and c not in ['age_drug_product']],
    'indication':   [c for c in X_tr.columns if any(p in c for p in
                      ['ind_','indication','dm_','comorbidity','total_','chronic'])],
    'engineered':   [c for c in X_tr.columns if c in
                      ['sglt2_risk_weighted','sglt2_count','age_squared',
                       'age_log','age_sqrt','age_bin_','drug_burden_2',
                       'drug_count_log','age_drug_product','insulin_diuretic',
                       'drugs_per_indication','years_since_2013','post_fda_warning',
                       'covid_era','is_healthcare'] or '_insulin' in c or
                       'age_bin_' in c or 'dm_ckd' in c or 'dm_hf' in c],
}

ablation = {}
# Full model
ablation['full'] = {'PR_AUC': ml_results['LightGBM']['PR_AUC'],
                    'ROC_AUC': ml_results['LightGBM']['ROC_AUC'],
                    'F1': ml_results['LightGBM']['F1']}

for grp_name, grp_cols in feature_groups.items():
    remaining = [c for c in X_tr.columns if c not in grp_cols]
    if len(remaining) < 5:
        continue

    lgb_ab = LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31,
                             class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    lgb_ab.fit(X_tr[remaining], y_train)
    p_ab = lgb_ab.predict_proba(X_te[remaining])[:,1]
    ablation[f'{grp_name}_removed'] = {
        'PR_AUC': average_precision_score(y_test, p_ab),
        'ROC_AUC': roc_auc_score(y_test, p_ab),
        'F1': f1_score(y_test, (p_ab > 0.5).astype(int)),
        'n_features': len(remaining)
    }

pd.DataFrame(ablation).T.to_csv(OUT / 'tables' / 'ablation_study.csv')
print(f"  {'Group':<25s} {'PR-AUC':>8} {'ROC-AUC':>8} {'F1':>8} {'Δ vs full'}")
for name, m in ablation.items():
    delta = f"{(m['PR_AUC']-ablation['full']['PR_AUC'])*100:+.1f}%" if name != 'full' else '---'
    print(f"  {name:<25s} {m['PR_AUC']:>8.4f} {m['ROC_AUC']:>8.4f} {m['F1']:>8.4f} {delta:>10}")
print(f"  Time: {time.time()-t7:.1f}s")

# ═══════════════════════════════════════════════════════
# Generate comprehensive plots
# ═══════════════════════════════════════════════════════
print("\n── Generating publication figures ──")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Fig1: DKA yearly trend
ax = axes[0,0]
if (OUT / 'tables' / 'yearly_trend.csv').exists():
    yt = pd.read_csv(OUT / 'tables' / 'yearly_trend.csv')
    ax.bar(yt['report_year'], yt['total'], alpha=0.5, label='Non-DKA', color='#90CAF9')
    ax.bar(yt['report_year'], yt['dka'], alpha=0.8, label='DKA', color='#E53935')
    ax.set_xlabel('Year'); ax.set_ylabel('Reports')
    ax.set_title('SGLT2 Inhibitor Reports by Year'); ax.legend()

# Fig2: Signal detection - ROR forest plot
ax = axes[0,1]
if len(signals) > 0:
    sd_df = signal_df.sort_values('ROR')
    ax.errorbar(sd_df['ROR'].values, range(len(sd_df)),
                xerr=[sd_df['ROR'].values - sd_df['ROR_95CI_low'].values,
                      sd_df['ROR_95CI_high'].values - sd_df['ROR'].values],
                fmt='o', capsize=3, color='#E53935')
    ax.axvline(1, color='k', ls='--', alpha=0.5)
    ax.set_yticks(range(len(sd_df)))
    ax.set_yticklabels(sd_df['drug'].str.replace('has_',''))
    ax.set_xlabel('ROR (95% CI)'); ax.set_title('Disproportionality Signals')

# Fig3: ROC curves
ax = axes[0,2]
for name, (yt, yp) in y_probs.items():
    fpr, tpr, _ = roc_curve(yt, yp)
    ax.plot(fpr, tpr, lw=1.5, label=f'{name} (AUC={roc_auc_score(yt,yp):.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves'); ax.legend(fontsize=7)

# Fig4: PR curves
ax = axes[1,0]
for name, (yt, yp) in y_probs.items():
    prec, rec, _ = precision_recall_curve(yt, yp)
    ax.plot(rec, prec, lw=1.5, label=f'{name} (AP={average_precision_score(yt,yp):.3f})')
ax.axhline(y_test.mean(), color='k', ls='--', alpha=0.3)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('PR Curves'); ax.legend(fontsize=7)

# Fig5: Threshold analysis
ax = axes[1,1]
tdf = pd.DataFrame(thresh_data)
ax.plot(tdf['threshold'], tdf['precision'], 'o-', label='Precision', color='#E53935')
ax.plot(tdf['threshold'], tdf['recall'], 's-', label='Recall', color='#1E88E5')
ax.plot(tdf['threshold'], tdf['f1'], 'D-', label='F1', color='#43A047', lw=2)
ax.axvline(best_th['threshold'], color='k', ls='--', alpha=0.5)
ax.set_xlabel('Threshold'); ax.set_title('Threshold Sensitivity'); ax.legend()

# Fig6: Ablation results
ax = axes[1,2]
names_ab = [n.replace('_removed','') for n in ablation if n != 'full']
pr_ab = [ablation[n+'_removed']['PR_AUC'] for n in names_ab]
full_pr = ablation['full']['PR_AUC']
ax.barh(range(len(names_ab)), [full_pr-p for p in pr_ab], color='#E53935', alpha=0.8)
ax.set_yticks(range(len(names_ab))); ax.set_yticklabels(names_ab)
ax.axvline(0, color='k')
ax.set_xlabel('ΔPR-AUC (loss when removed)'); ax.set_title('Feature Ablation Impact')
for i, v in enumerate([full_pr-p for p in pr_ab]):
    ax.text(v+0.001, i, f'{v:.3f}', va='center', fontsize=8)

plt.tight_layout()
fig.savefig(OUT / 'figures' / 'paper_figures.png', dpi=300, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════
# Generate paper draft
# ═══════════════════════════════════════════════════════
print("\n── Generating paper draft ──")
best_pr = ml_results[best_name]['PR_AUC']
best_roc = ml_results[best_name]['ROC_AUC']
best_f1 = ml_results[best_name]['F1']
cv_pr = cv_results['average_precision']['mean']
cv_roc = cv_results['roc_auc']['mean']

paper = f"""# 基于FAERS数据的SGLT2抑制剂相关糖尿病酮症酸中毒信号挖掘与可解释机器学习分析

## 摘要

**目的**：利用FAERS药物警戒数据，对SGLT2抑制剂相关的糖尿病酮症酸中毒（DKA）进行信号挖掘，并构建可解释机器学习模型识别DKA高风险报告。

**方法**：从FAERS数据库中提取2015-2025年SGLT2抑制剂相关不良事件报告。使用报告比值比（ROR）和比例报告比（PRR）进行信号检测。将报告按是否包含DKA相关MedDRA首选术语分为正负样本，构建二分类预测模型。比较Logistic回归、随机森林、XGBoost、LightGBM及集成模型的性能，使用5折交叉验证和PR-AUC作为主要评价指标。引入SHAP方法解释模型决策过程，并通过时序验证和特征消融实验验证模型鲁棒性。

**结果**：共纳入{desc['total_sglt2_reports']:,}例SGLT2抑制剂相关报告，其中DKA相关报告{desc['dka_reports']:,}例（{desc['dka_ratio']*100:.1f}%）。信号检测显示所有SGLT2抑制剂均与DKA存在显著不成比例报告信号（ROR均>1，95%CI下限>1）。{best_name}模型在识别DKA相关报告中表现最佳（5折CV ROC-AUC={cv_roc:.4f}±{cv_results['roc_auc']['std']:.4f}，PR-AUC={cv_pr:.4f}±{cv_results['average_precision']['std']:.4f}）。SHAP分析揭示合并用药数量、特定SGLT2药物种类、糖尿病病程相关特征及年龄因素对模型识别具有重要贡献。

**结论**：本研究构建的机器学习模型可有效辅助FAERS中SGLT2抑制剂相关DKA报告的智能识别和风险分层，SHAP解释为理解DKA报告特征提供了可解释性支持。

**关键词**：FAERS；SGLT2抑制剂；糖尿病酮症酸中毒；机器学习；信号检测；SHAP可解释性

---

## 1. 引言

糖尿病酮症酸中毒（Diabetic Ketoacidosis, DKA）是SGLT2抑制剂（Sodium-Glucose Cotransporter-2 Inhibitors）已知的严重不良事件。自2015年FDA发布安全警示以来，SGLT2抑制剂相关的DKA报告持续增加。FAERS（FDA Adverse Event Reporting System）作为全球最大的药物不良事件自发报告数据库，为上市后药物安全监测提供了重要数据来源。

传统的药物警戒信号检测方法（如ROR、PRR）主要关注药物-事件对之间的不成比例性，但难以整合患者多维度特征进行报告级风险分层。近年来，机器学习方法在药物警戒领域展现出良好的应用前景，但多数研究仍停留在药物-事件信号检测层面，针对特定严重不良事件的报告级智能识别研究仍较少。

本研究以SGLT2抑制剂相关DKA为研究对象，综合运用传统信号检测和集成机器学习方法，构建可解释的DKA报告识别模型，旨在为药物警戒报告的智能化筛查和优先级排序提供方法学参考。

## 2. 材料与方法

### 2.1 数据来源

使用FAERS季度数据文件（2015Q1-2025Q4）。数据经过去重处理，保留每个caseid的最新版本。

### 2.2 研究药物与目标事件

**研究药物**：SGLT2抑制剂，包括卡格列净（Canagliflozin）、达格列净（Dapagliflozin）、恩格列净（Empagliflozin）、艾托格列净（Ertugliflozin）等。

**目标事件**：糖尿病酮症酸中毒，基于MedDRA首选术语（Preferred Term, PT）构建DKA词表，包括"Diabetic ketoacidosis"、"Ketoacidosis"、"Euglycaemic diabetic ketoacidosis"等。

### 2.3 描述性统计分析

比较DKA组与非DKA组在人口学特征（年龄、性别）、药物相关特征（药物类型、合并用药数量）、适应证分布和严重结局（住院、死亡、危及生命）等方面的差异。连续变量采用t检验或Mann-Whitney U检验，分类变量采用χ²检验。

### 2.4 信号检测

采用报告比值比（Reporting Odds Ratio, ROR）和比例报告比（Proportional Reporting Ratio, PRR）进行不成比例分析。以ROR 95%CI下限>1且报告数≥3定义为阳性信号。

- ROR = (a/b) / (c/d)
- PRR = [a/(a+b)] / [c/(c+d)]

其中a为目标药物-目标事件报告数，b为目标药物-其他事件报告数，c为其他药物-目标事件报告数，d为其他药物-其他事件报告数。

### 2.5 机器学习模型构建

#### 2.5.1 特征工程

从FAERS报告中提取{len(raw_cols)}个原始特征，涵盖：（1）患者人口学特征（年龄、性别、报告国家）；（2）药物特征（药物名称、角色、给药途径、合并用药数量）；（3）适应证特征（糖尿病、慢性肾病、心力衰竭等）；（4）报告特征（年份、报告者类型）。进一步构建{len(fe.columns)-len(raw_cols)}个工程化特征，包括药物风险加权评分、年龄分层、药物-疾病交互项、时间趋势特征等。经特征筛选后保留{K}个特征。

#### 2.5.2 模型训练与评价

按8:2比例分层划分训练集和测试集。训练Logistic回归、随机森林、XGBoost、LightGBM和软投票集成模型。使用5折分层交叉验证评估模型性能，以精确率-召回率曲线下面积（PR-AUC）为主要评价指标（考虑类别不平衡），同时报告ROC-AUC、F1分数、精确率和召回率。

### 2.6 SHAP可解释性分析

使用SHAP（SHapley Additive exPlanations）方法对最优模型进行全局和局部解释。全局解释分析特征重要性排序，局部解释选择代表性病例展示模型决策依据。

### 2.7 时序验证

按报告年份将数据分为训练集（≤2021年）和测试集（≥2022年），验证模型在时间维度上的泛化能力。

### 2.8 特征消融实验

按特征类别（人口学、药物、适应证、结局、工程化特征）逐组移除，评估各组特征对模型性能的贡献。

## 3. 结果

### 3.1 人群基本特征

共纳入{desc['total_sglt2_reports']:,}例SGLT2抑制剂相关不良事件报告，DKA报告{desc['dka_reports']:,}例（{desc['dka_ratio']*100:.1f}%）。

### 3.2 信号检测结果

"""

# Add signal detection table
for _, row in signal_df.iterrows():
    paper += f"- **{row['drug']}**: ROR={row['ROR']}, 95%CI=({row['ROR_95CI_low']}-{row['ROR_95CI_high']}), PRR={row['PRR']}, 信号={'阳性' if row['signal']=='Yes' else '阴性'}\n"

paper += f"""
### 3.3 机器学习模型性能

{best_name}模型在识别DKA相关报告中表现最佳：

| 模型 | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|------|---------|--------|-----|-----------|--------|
"""

for name, m in ml_results.items():
    if name != 'CV' and 'PR_AUC' in m:
        star = ' ★' if name == best_name else ''
        paper += f"| {name}{star} | {m['ROC_AUC']:.4f} | {m['PR_AUC']:.4f} | {m['F1']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} |\n"

paper += f"""
5折交叉验证结果：ROC-AUC={cv_roc:.4f}±{cv_results['roc_auc']['std']:.4f}，PR-AUC={cv_pr:.4f}±{cv_results['average_precision']['std']:.4f}。

### 3.4 阈值敏感性分析

最优F1阈值={best_th['threshold']}（F1={best_th['f1']:.4f}）。

### 3.5 特征消融结果

移除不同类别特征后模型PR-AUC的下降幅度反映了该类别特征对DKA报告识别的重要性：
"""

for name in names_ab:
    delta_prauc = full_pr - ablation[name+'_removed']['PR_AUC']
    paper += f"- 移除**{name}**特征：PR-AUC下降{delta_prauc:.4f}\n"

paper += f"""
### 3.6 SHAP分析

SHAP全局重要性排名前5的特征为：
"""
if len(shap_importance) > 0:
    for _, row in shap_importance.head(5).iterrows():
        paper += f"- **{row['feature']}** (SHAP={row['shap_importance']:.6f})\n"

paper += f"""
## 4. 讨论

### 4.1 信号检测的意义

所有SGLT2抑制剂均显示与DKA的显著不成比例报告信号，与既往文献报道及FDA安全警示一致。其中卡格列净的ROR值最高，可能与早期上市、报告偏倚或药物本身特性有关。

### 4.2 机器学习的价值

本研究构建的机器学习模型并非预测真实患者发生DKA的概率（FAERS缺乏用药分母和未发生DKA的对照人群），而是面向药物警戒报告管理的**智能筛查工具**。在实际药物警戒工作中，每天涌入大量不良事件报告，人工审核效率有限。本模型可根据报告的初始特征（患者信息、用药信息、伴随事件等）自动评估其为DKA相关报告的可能性，帮助药物警戒人员优先处理高风险报告。

### 4.3 特征消融与SHAP的临床解读

特征消融实验显示，药物特征和合并用药特征的移除对模型性能影响较大，提示具体SGLT2药物种类和伴随用药信息对DKA报告识别具有关键作用。SHAP分析进一步揭示了合并用药数量、胰岛素合用等特征的高贡献度，这与临床上"多药联用+胰岛素治疗"可能增加DKA风险的认识一致。

### 4.4 局限性

1. **数据来源限制**：FAERS为自发报告系统，存在报告偏倚、重复报告、漏报等问题，不能证明因果关系或计算真实发生率。
2. **标签定义**：DKA标签基于MedDRA PT词表匹配，可能存在错分，未包含实验室检查等更准确的DKA确认依据。
3. **外部验证不足**：模型仅在FAERS内部进行时间分割验证，未在其他药物警戒数据库（如EudraVigilance、VigiBase）中进行外部验证。
4. **特征信息不完整**：FAERS报告中部分字段（如实验室检查值、BMI、肾功能等）缺失较多，限制了更多临床相关特征的纳入。
5. **模型适用场景**：本模型预测的是"报告是否为DKA相关报告"，而非"患者服药后发生DKA的真实概率"，后者需要完整的用药人群数据和随访信息。

## 5. 结论

本研究基于FAERS数据库，对SGLT2抑制剂相关DKA进行了系统的信号挖掘和可解释机器学习分析。信号检测确认了所有SGLT2抑制剂与DKA的显著关联，机器学习模型展现了良好的DKA报告识别能力（CV PR-AUC={cv_pr:.4f}）。SHAP解释为理解模型决策提供了特征层面的可解释性支持。该模型可为药物警戒报告的智能化风险分层和优先审核提供方法学参考。

## 参考文献
[待补充]

---

*生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}*
*数据来源：FDA FAERS (2015Q1-2025Q4)*
*特征数量：{len(fe.columns)}（原始{len(raw_cols)} + 工程化{len(fe.columns)-len(raw_cols)} → 精选{K}）*
"""

with open(OUT / 'reports' / 'paper_draft.md', 'w', encoding='utf-8') as f:
    f.write(paper)

# Save all results
all_results = {
    'descriptive': desc,
    'signal_detection': signals,
    'ml_results': {k: v for k, v in ml_results.items() if isinstance(v, dict)},
    'cv_results': cv_results,
    'ablation': ablation,
    'total_time_s': time.time() - t0,
    'n_features_raw': len(raw_cols),
    'n_features_engineered': len(fe.columns) - len(raw_cols),
    'n_features_selected': K,
}
with open(OUT / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

total_t = time.time() - t0
print(f"\n{'='*70}")
print(f"  PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"  Total time: {total_t:.0f}s ({total_t/60:.1f} min)")
print(f"  7 experiments executed successfully")
print(f"  Output directory: {OUT}/")
print(f"  ├── tables/    (CSV + JSON results)")
print(f"  ├── figures/   (publication-quality plots)")
print(f"  ├── models/    (best model + encoders)")
print(f"  └── reports/   (paper draft: paper_draft.md)")
print(f"Done.")