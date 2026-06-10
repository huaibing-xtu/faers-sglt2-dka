"""
生成论文所有图表 (Times New Roman, 30号字, 300ppi)
输出PNG和PDF两种格式

图表列表:
1. Figure 1: 年度趋势图 (DKA报告比例)
2. Figure 2: 信号检测森林图 (ROR)
3. Figure 3: ROC and PR Curves (merged)
4. Figure 4: 阈值敏感性分析
5. Figure 5: 特征消融分析
6. SHAP Summary Plot
7. SHAP Dependence Top 10
8. 时序验证结果

注意: 已删除所有标题，图注位置已优化避免遮挡内容
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 16  # 图注字号调整为16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

ROOT = Path(r'E:\FAERS_DKA')
DATA = ROOT / 'data' / 'processed'
OUT  = ROOT / 'outputs' / 'paper_results' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  GENERATING PAPER FIGURES (NO TITLES, OPTIMIZED LEGENDS)")
print("=" * 70)

# ──────── Load Data ────────
print("\n1. Loading data...")
df = pd.read_parquet(DATA / "model_dataset.parquet")
sglt2 = df[df["has_study_drug_any"].eq(1)].copy()
sglt2["label_target_event"] = sglt2["label_target_event"].fillna(0).astype(int)

# Load results
signal_df = pd.read_csv(ROOT / 'outputs' / 'paper_results' / 'tables' / 'signal_detection.csv')
ablation_df = pd.read_csv(ROOT / 'outputs' / 'paper_results' / 'tables' / 'ablation_study.csv', index_col=0)
temporal_df = pd.read_csv(ROOT / 'outputs' / 'paper_results' / 'tables' / 'temporal_validation.csv')
threshold_df = pd.read_csv(ROOT / 'outputs' / 'paper_results' / 'tables' / 'threshold_analysis.csv')
shap_df = pd.read_csv(ROOT / 'outputs' / 'paper_results' / 'tables' / 'shap_importance.csv')

# ──────── Feature Engineering (same as pipeline) ────────
print("2. Preparing features...")
EXCLUDE_PREFIX = {'any_serious', 'outcome_'}
EXCLUDE_EXACT  = {'primaryid', 'caseid', 'caseversion', 'quarter',
                  'label_target_event', 'has_study_drug_any',
                  'fda_dt', 'fda_dt_parsed', 'event_dt', 'init_fda_dt', 'init_fda_dt_parsed'}

def s(x):
    return pd.to_numeric(x, errors='coerce').fillna(0)

raw_cols = [c for c in sglt2.columns
            if not any(c.startswith(p) for p in EXCLUDE_PREFIX) and c not in EXCLUDE_EXACT]
fe = sglt2[raw_cols].copy()

for c in ['age_num', 'drug_count', 'indication_count', 'report_year']:
    if c in fe.columns: fe[c] = s(fe[c])

for c in fe.columns:
    if any(c.startswith(p) for p in ['has_', 'concomitant_', 'ind_']):
        fe[c] = s(fe[c]).astype(int)

for c in fe.select_dtypes(include=['object']).columns:
    fe[c] = pd.Categorical(fe[c]).codes.astype(float)

# Engineered features
sglt2_drugs = [c for c in fe.columns if c.startswith('has_')]
concomitants = [c for c in fe.columns if c.startswith('concomitant_')]
indications = [c for c in fe.columns if c.startswith('ind_')]

risk_w = {'has_canagliflozin':1.5,'has_empagliflozin':1.0,'has_dapagliflozin':0.8,
          'has_ertugliflozin':0.7,'has_ipragliflozin':0.5,'has_luseogliflozin':0.5,
          'has_sotagliflozin':0.6,'has_tofogliflozin':0.5}
fe['sglt2_risk_weighted'] = sum(fe.get(c,0)*w for c,w in risk_w.items())
fe['sglt2_count'] = fe[sglt2_drugs].sum(axis=1) if sglt2_drugs else 0

if 'age_num' in fe.columns:
    a = s(fe['age_num'])
    fe['age_squared'] = a**2; fe['age_log'] = np.log1p(a); fe['age_sqrt'] = np.sqrt(a.clip(0))
    for lo, hi, nm in [(0,18,'lt18'),(18,45,'18_44'),(45,65,'45_64'),(65,75,'65_74'),(75,200,'75plus')]:
        fe[f'age_bin_{nm}'] = ((a>=lo)&(a<hi)).astype(int)

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

if 'report_year' in fe.columns:
    ry = s(fe['report_year'])
    fe['years_since_2013'] = (ry-2013).clip(0)
    fe['post_fda_warning'] = (ry>=2015).astype(int)
    fe['covid_era'] = (ry>=2020).astype(int)
if 'reporter_type' in fe.columns:
    fe['is_healthcare'] = fe['reporter_type'].isin([0,1,2]).astype(int)

fe = fe.select_dtypes(include=['int64','float64','int32','float32']).fillna(0)
fe = fe.replace([np.inf,-np.inf], 0)
const_cols = [c for c in fe.columns if fe[c].nunique()<=1]
if const_cols: fe = fe.drop(columns=const_cols)

X = fe; y = sglt2["label_target_event"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = RobustScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
K = min(80, X_train_s.shape[1])
selector = SelectKBest(f_classif, k=K)
X_tr = pd.DataFrame(selector.fit_transform(X_train_s, y_train),
                    columns=X_train_s.columns[selector.get_support()])
X_te = pd.DataFrame(selector.transform(X_test_s),
                    columns=X_train_s.columns[selector.get_support()])

# Train model
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7,
                     class_weight='balanced', min_child_samples=50,
                     subsample=0.85, colsample_bytree=0.85,
                     reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
lgb.fit(X_tr, y_train)
y_prob_lgb = lgb.predict_proba(X_te)[:, 1]

from xgboost import XGBClassifier
pos_w = (y_train==0).sum() / max((y_train==1).sum(), 1)
xgb_mod = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        scale_pos_weight=pos_w, subsample=0.85, colsample_bytree=0.85,
                        random_state=42, n_jobs=-1)
xgb_mod.fit(X_tr, y_train)
y_prob_xgb = xgb_mod.predict_proba(X_te)[:, 1]

# Ensemble
y_prob_ens = (y_prob_lgb + y_prob_xgb) / 2

print(f"   Features: {X.shape[1]} raw → {K} selected")

# ═══════════════════════════════════════════════════════
# FIGURE 1: Annual Trend (DKA Reports)
# ═══════════════════════════════════════════════════════
print("\n3. Generating Figure 1: Annual Trend...")
yearly = sglt2.groupby('report_year').agg(
    total=('label_target_event', 'count'),
    dka=('label_target_event', 'sum')
).reset_index()
yearly['dka_pct'] = yearly['dka'] / yearly['total'] * 100

# 删除2013年的数据（实验中未使用）
yearly = yearly[yearly['report_year'] >= 2015]

fig, ax1 = plt.subplots(figsize=(12, 8))

# Bar chart for total reports
color_total = '#90CAF9'
color_dka = '#E53935'

ax1.bar(yearly['report_year'], yearly['total'], alpha=0.7, color=color_total, label='Non-DKA Reports', width=0.8)
ax1.bar(yearly['report_year'], yearly['dka'], alpha=0.9, color=color_dka, label='DKA Reports', width=0.8)

ax1.set_xlabel('Year', fontsize=30, fontname='Times New Roman')
ax1.set_ylabel('Number of Reports', fontsize=30, fontname='Times New Roman')
ax1.tick_params(axis='both', labelsize=25)
ax1.set_xticks(yearly['report_year'])
ax1.set_xticklabels(yearly['report_year'].astype(int), rotation=45, ha='right')

# Secondary axis for percentage
ax2 = ax1.twinx()
ax2.plot(yearly['report_year'], yearly['dka_pct'], 'o-', color='darkred', linewidth=2, markersize=8, label='DKA Proportion (%)')
ax2.set_ylabel('DKA Proportion (%)', fontsize=30, fontname='Times New Roman')
ax2.tick_params(axis='y', labelsize=25)

# Combined legend - 放在左上角，避免遮挡右侧内容
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=16, framealpha=0.9)

# 不设置标题
plt.tight_layout()

# Save PNG and PDF
plt.savefig(OUT / 'figure1_annual_trend.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig(OUT / 'figure1_annual_trend.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"   Saved: figure1_annual_trend.png/pdf")

# ═══════════════════════════════════════════════════════
# FIGURE 2: Signal Detection Forest Plot (ROR)
# ═══════════════════════════════════════════════════════
print("\n4. Generating Figure 2: Signal Detection Forest Plot...")
fig, ax = plt.subplots(figsize=(12, 8))

# Sort by ROR
sd_df = signal_df.sort_values('ROR', ascending=True)

# Plot
y_pos = range(len(sd_df))
ax.errorbar(sd_df['ROR'].values, y_pos,
            xerr=[sd_df['ROR'].values - sd_df['ROR_95CI_low'].values,
                  sd_df['ROR_95CI_high'].values - sd_df['ROR'].values],
            fmt='o', capsize=5, capthick=2, color='#E53935', markersize=10, linewidth=2)

ax.axvline(1, color='black', linestyle='--', alpha=0.5, linewidth=2)
ax.set_yticks(y_pos)
ax.set_yticklabels([d.replace('has_', '') for d in sd_df['drug']], fontsize=25)
ax.set_xlabel('Reporting Odds Ratio (95% CI)', fontsize=30, fontname='Times New Roman')
ax.tick_params(axis='x', labelsize=25)

# Add ROR values as text
for i, (ror, low, high) in enumerate(zip(sd_df['ROR'], sd_df['ROR_95CI_low'], sd_df['ROR_95CI_high'])):
    ax.text(high + 2, i, f'{ror:.1f} ({low:.1f}-{high:.1f})', va='center', fontsize=18)

# 不设置标题
plt.tight_layout()
plt.savefig(OUT / 'figure2_signal_detection.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig(OUT / 'figure2_signal_detection.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"   Saved: figure2_signal_detection.png/pdf")

# ═══════════════════════════════════════════════════════
# FIGURE 3: ROC + PR Curves (merged)
# ═══════════════════════════════════════════════════════
print("\n5. Generating Figure 3: ROC and PR Curves...")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# ── Left: ROC curves ──
ax = axes[0]
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_prob_lgb)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ens)

auc_lgb = auc(fpr_lgb, tpr_lgb)
auc_xgb = auc(fpr_xgb, tpr_xgb)
auc_ens = auc(fpr_ens, tpr_ens)

ax.plot(fpr_lgb, tpr_lgb, color='#1E88E5', linewidth=2.5, label=f'LightGBM (AUC = {auc_lgb:.3f})')
ax.plot(fpr_xgb, tpr_xgb, color='#43A047', linewidth=2.5, label=f'XGBoost (AUC = {auc_xgb:.3f})')
ax.plot(fpr_ens, tpr_ens, color='#E53935', linewidth=2.5, label=f'Ensemble (AUC = {auc_ens:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random (AUC = 0.500)')

ax.set_xlabel('False Positive Rate', fontsize=28, fontname='Times New Roman')
ax.set_ylabel('True Positive Rate', fontsize=28, fontname='Times New Roman')
ax.legend(loc='lower right', fontsize=14, framealpha=0.9)
ax.tick_params(axis='both', labelsize=22)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

# ── Right: PR curves ──
ax = axes[1]
prec_lgb, rec_lgb, _ = precision_recall_curve(y_test, y_prob_lgb)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)
prec_ens, rec_ens, _ = precision_recall_curve(y_test, y_prob_ens)

ap_lgb = average_precision_score(y_test, y_prob_lgb)
ap_xgb = average_precision_score(y_test, y_prob_xgb)
ap_ens = average_precision_score(y_test, y_prob_ens)

ax.plot(rec_lgb, prec_lgb, color='#1E88E5', linewidth=2.5, label=f'LightGBM (AP = {ap_lgb:.3f})')
ax.plot(rec_xgb, prec_xgb, color='#43A047', linewidth=2.5, label=f'XGBoost (AP = {ap_xgb:.3f})')
ax.plot(rec_ens, prec_ens, color='#E53935', linewidth=2.5, label=f'Ensemble (AP = {ap_ens:.3f})')

# No-skill baseline
prevalence = y_test.mean()
ax.axhline(y=prevalence, color='black', linestyle='--', linewidth=2, alpha=0.5, label=f'No-skill (Prevalence = {prevalence:.3f})')

ax.set_xlabel('Recall', fontsize=28, fontname='Times New Roman')
ax.set_ylabel('Precision', fontsize=28, fontname='Times New Roman')
ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
ax.tick_params(axis='both', labelsize=22)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'figure3_roc_pr_curves.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig(OUT / 'figure3_roc_pr_curves.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"   Saved: figure3_roc_pr_curves.png/pdf")

# ═══════════════════════════════════════════════════════
# FIGURE 5: Threshold Sensitivity Analysis (优化图注位置)
# ═══════════════════════════════════════════════════════
print("\n6. Generating Figure 4: Threshold Sensitivity...")
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', color='#E53935', linewidth=2.5, markersize=8, label='Precision')
ax.plot(threshold_df['threshold'], threshold_df['recall'], 's-', color='#1E88E5', linewidth=2.5, markersize=8, label='Recall')
ax.plot(threshold_df['threshold'], threshold_df['f1'], 'D-', color='#43A047', linewidth=3, markersize=10, label='F1 Score')

# Find optimal F1 threshold
opt_idx = threshold_df['f1'].idxmax()
opt_threshold = threshold_df.loc[opt_idx, 'threshold']
opt_f1 = threshold_df.loc[opt_idx, 'f1']
ax.axvline(x=opt_threshold, color='black', linestyle='--', linewidth=2, alpha=0.5)

# 将标注放在图的左上角，避免遮挡曲线
ax.annotate(f'Optimal F1 = {opt_f1:.3f}\n(Threshold = {opt_threshold:.2f})',
            xy=(opt_threshold, opt_f1), xytext=(0.05, 0.95),
            xycoords='axes fraction', textcoords='axes fraction',
            fontsize=14, arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
            verticalalignment='top')

ax.set_xlabel('Classification Threshold', fontsize=30, fontname='Times New Roman')
ax.set_ylabel('Score', fontsize=30, fontname='Times New Roman')

# 图注放在图外右侧，避免遮挡曲线
ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=16, framealpha=0.9)

ax.tick_params(axis='both', labelsize=25)
ax.set_xlim([0.05, 1.0])
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)

# 不设置标题
plt.tight_layout()
plt.savefig(OUT / 'figure4_threshold_analysis.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig(OUT / 'figure4_threshold_analysis.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"   Saved: figure4_threshold_analysis.png/pdf")

# ═══════════════════════════════════════════════════════
# FIGURE 6: Feature Ablation Analysis
# ═══════════════════════════════════════════════════════
print("\n8. Generating Figure 5: Feature Ablation...")
fig, ax = plt.subplots(figsize=(10, 8))

# Prepare data
ablation_data = ablation_df[ablation_df.index != 'full'].copy()
full_prauc = ablation_df.loc['full', 'PR_AUC']
ablation_data['delta'] = full_prauc - ablation_data['PR_AUC']
ablation_data['group'] = ablation_data.index.str.replace('_removed', '')

# Sort by delta
ablation_data = ablation_data.sort_values('delta', ascending=True)

colors = ['#E53935' if d > 0.05 else '#FFCDD2' for d in ablation_data['delta']]

bars = ax.barh(range(len(ablation_data)), ablation_data['delta'], color=colors, alpha=0.8, height=0.6)
ax.set_yticks(range(len(ablation_data)))
ax.set_yticklabels(ablation_data['group'], fontsize=25)
ax.axvline(x=0, color='black', linewidth=2)
ax.set_xlabel('ΔPR-AUC (Performance Loss When Removed)', fontsize=30, fontname='Times New Roman')
ax.tick_params(axis='x', labelsize=25)

# Add value labels
for i, (idx, row) in enumerate(ablation_data.iterrows()):
    ax.text(row['delta'] + 0.001, i, f'{row["delta"]:.4f}', va='center', fontsize=20)

# 不设置标题
plt.tight_layout()
plt.savefig(OUT / 'figure6_ablation_analysis.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig(OUT / 'figure6_ablation_analysis.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"   Saved: figure6_ablation_analysis.png/pdf")

# ═══════════════════════════════════════════════════════
# FIGURE 7: SHAP Summary Plot (使用缩写)
# ═══════════════════════════════════════════════════════
print("\n9. Generating Figure 7: SHAP Summary Plot...")
try:
    import shap
    X_explain = X_te.sample(min(2000, len(X_te)), random_state=42)
    explainer = shap.TreeExplainer(lgb)
    shap_values = explainer.shap_values(X_explain)

    # 特征名称缩写映射
    feature_abbrev = {
        'occp_cod': 'OccCod',
        'drug_count': 'DrgCnt',
        'ind_diabetes': 'IndDM',
        'report_year': 'RptYr',
        'occr_country': 'OccCtry',
        'indication_count': 'IndCnt',
        'reporter_country': 'RepCtry',
        'study_drug_route': 'DrgRte',
        'concomitant_diuretic': 'ConDiur',
        'age_num': 'Age',
        'concomitant_insulin': 'ConIns',
        'sglt2_risk_weighted': 'SGLT2Risk',
        'wt': 'Wt',
        'drugs_per_indication': 'DrgPerInd',
        'reaction_count': 'ReactCnt',
        'concomitant_metformin': 'ConMet',
        'sex': 'Sex',
        'age_drug_product': 'AgeDrg',
        'has_empagliflozin': 'Empa',
        'concomitant_total': 'ConTot',
        'age_grp': 'AgeGrp',
        'drug_burden_2': 'DrgBrd2',
        'years_since_2013': 'Yrs2013',
        'comorbidity_score': 'ComScor',
        'age_bin_18_44': 'Age18-44',
        'age_bin_45_64': 'Age45-64',
        'total_indications': 'TotInd',
        'age_squared': 'Age²',
        'has_dapagliflozin': 'Dapa',
        'concomitant_steroid': 'ConSter',
        'age_cod': 'AgeCod',
        'has_canagliflozin': 'Cana',
        'study_drug_main': 'DrgMain',
        'age_bin_75plus': 'Age75+',
        'ind_chronic_kidney_disease': 'IndCKD',
        'reporter_type': 'RepType',
        'ind_heart_failure': 'IndHF',
        'drug_count_log': 'DrgLog',
        'concomitant_nsaid': 'ConNSAID',
        'wt_cod': 'WtCod',
        'has_empagliflozin_insulin': 'EmpIns',
        'covid_era': 'Covid',
        'has_dapagliflozin_insulin': 'DapIns',
        'has_canagliflozin_insulin': 'CanIns',
        'age_log': 'AgeLog',
        'age_bin_65_74': 'Age65-74',
        'insulin_diuretic': 'InsDiur',
        'dm_hf': 'DM-HF',
        'dm_ckd': 'DM-CKD',
        'has_sotagliflozin': 'Sota',
        'has_ertugliflozin_insulin': 'ErtIns',
        'sglt2_count': 'SGLT2Cnt',
        'has_tofogliflozin': 'Tofog',
        'has_luseogliflozin': 'Luseo',
        'has_ertugliflozin': 'Ertu',
        'is_healthcare': 'HlthCr',
        'has_ipragliflozin': 'Ipra',
        'study_drug_role': 'DrgRole',
        'age_sqrt': 'AgeSqrt',
        'age_bin_lt18': 'Age<18',
        'post_fda_warning': 'PostFDA',
    }

    # 创建缩写后的数据副本
    X_explain_abbr = X_explain.copy()
    X_explain_abbr.columns = [feature_abbrev.get(c, c) for c in X_explain.columns]

    # Create SHAP summary plot
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_explain_abbr, show=False, max_display=15, plot_size=None)

    # 设置坐标轴标签和字号
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=30, fontname='Times New Roman')
    plt.ylabel('Feature', fontsize=30, fontname='Times New Roman')

    # 调整Y轴刻度字号为24
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=24)

    # 不设置标题
    plt.tight_layout()
    plt.savefig(OUT / 'figure7_shap_summary.png', dpi=300, bbox_inches='tight', format='png')
    plt.savefig(OUT / 'figure7_shap_summary.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"   Saved: figure7_shap_summary.png/pdf")
except Exception as e:
    print(f"   SHAP summary plot failed: {e}")

# ═══════════════════════════════════════════════════════
# FIGURE 8: Calibration Curve
# ═══════════════════════════════════════════════════════
print("\n10. Generating Figure 8: Calibration Curve...")
fig, ax = plt.subplots(figsize=(10, 8))

fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob_lgb, n_bins=10)

ax.plot(mean_predicted_value, fraction_of_positives, 's-', color='#E53935', linewidth=2.5, markersize=10, label='LightGBM')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfectly Calibrated')

brier = brier_score_loss(y_test, y_prob_lgb)
ax.text(0.05, 0.95, f'Brier Score = {brier:.4f}', transform=ax.transAxes,
        verticalalignment='top', fontsize=16, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Mean Predicted Probability', fontsize=30, fontname='Times New Roman')
ax.set_ylabel('Fraction of Positives', fontsize=30, fontname='Times New Roman')
ax.legend(loc='lower right', fontsize=16, framealpha=0.9)
ax.tick_params(axis='both', labelsize=25)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

# 不设置标题
plt.tight_layout()
plt.savefig(OUT / 'figure8_calibration_curve.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig(OUT / 'figure8_calibration_curve.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"   Saved: figure8_calibration_curve.png/pdf")

# ═══════════════════════════════════════════════════════
# FIGURE 9: Temporal Validation
# ═══════════════════════════════════════════════════════
print("\n11. Generating Figure 9: Temporal Validation...")
fig, ax = plt.subplots(figsize=(10, 8))

# Bar chart for temporal validation
x = np.arange(2)
width = 0.35

bars1 = ax.bar(x - width/2, temporal_df['PR_AUC'], width, label='PR-AUC', color='#E53935', alpha=0.8)
bars2 = ax.bar(x + width/2, temporal_df['ROC_AUC'], width, label='ROC-AUC', color='#1E88E5', alpha=0.8)

ax.set_xlabel('Dataset', fontsize=30, fontname='Times New Roman')
ax.set_ylabel('AUC Score', fontsize=30, fontname='Times New Roman')
ax.set_xticks(x)
ax.set_xticklabels(['Train (≤2021)', 'Test (≥2022)'], fontsize=25)
ax.legend(loc='upper left', fontsize=16, framealpha=0.9)
ax.tick_params(axis='y', labelsize=25)
ax.set_ylim([0.5, 1.05])

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=20)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontsize=20)

# 不设置标题
plt.tight_layout()
plt.savefig(OUT / 'figure9_temporal_validation.png', dpi=300, bbox_inches='tight', format='png')
plt.savefig(OUT / 'figure9_temporal_validation.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.close()
print(f"   Saved: figure9_temporal_validation.png/pdf")

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  FIGURE GENERATION COMPLETE (NO TITLES)")
print(f"{'='*70}")
print(f"  Output directory: {OUT}/")
print(f"  ├── figure1_annual_trend.png/pdf")
print(f"  ├── figure2_signal_detection.png/pdf")
print(f"  ├── figure3_roc_pr_curves.png/pdf")
print(f"  ├── figure4_threshold_analysis.png/pdf")
print(f"  ├── figure5_ablation_analysis.png/pdf")
print(f"  ├── figure6_shap_summary.png/pdf")
print(f"  ├── figure7_shap_dependence_top10_part1.png/pdf")
print(f"  ├── figure8_shap_dependence_top10_part2.png/pdf")
print(f"  └── figure9_temporal_validation.png/pdf")
print(f"\n  Format: Times New Roman font, 30pt axis labels, 300 DPI")
print(f"  Output: PNG + PDF formats")
print(f"  Note: No titles, legends optimized to avoid obstruction")
print(f"Done.")
