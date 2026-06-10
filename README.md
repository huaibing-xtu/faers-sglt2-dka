# FAERS SGLT2-DKA Analysis

> **SGLT2 Inhibitor-Associated Diabetic Ketoacidosis Signal Detection and Explainable Machine Learning Analysis using FAERS Database**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs40264--026--01630--x-lightgrey)](https://doi.org/10.1007/s40264-026-01630-x)
[![Stars](https://img.shields.io/github/stars/huaibing-xtu/faers-sglt2-dka?style=social)](https://github.com/huaibing-xtu/faers-sglt2-dka)
[![Forks](https://img.shields.io/github/forks/huaibing-xtu/faers-sglt2-dka?style=social)](https://github.com/huaibing-xtu/faers-sglt2-dka/forks)
[![Issues](https://img.shields.io/github/issues/huaibing-xtu/faers-sglt2-dka.svg)](https://github.com/huaibing-xtu/faers-sglt2-dka/issues)

## 📋 Overview

This project performs pharmacovigilance signal detection and explainable machine learning analysis of diabetic ketoacidosis (DKA) associated with sodium-glucose cotransporter-2 (SGLT2) inhibitors using the FDA Adverse Event Reporting System (FAERS) database.

### Key Features

1. **Signal Detection** - Reporting Odds Ratio (ROR) and Proportional Reporting Ratio (PRR) disproportionality analysis
2. **Machine Learning** - Ensemble LightGBM/XGBoost models for DKA report identification
3. **Explainability** - SHAP (SHapley Additive exPlanations) for interpretable model decisions
4. **Temporal Validation** - Time-based train/test split to assess generalizability
5. **Outcome-Leakage Control** - Models trained without post-event outcome variables

## 📁 Project Structure

```
faers_sglt2_dka/
├── README.md                      # This file
├── README-cn.md                   # Chinese version
├── setup.py                       # Package installation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── config/                        # Configuration files
│   └── terms.yml                  # MedDRA terms and drug definitions
├── src/                           # Source code
│   └── faers_sglt2_dka/          # Core package
│       ├── __init__.py
│       ├── download.py           # FAERS data download
│       ├── io.py                 # Data I/O utilities
│       ├── preprocess.py         # Data preprocessing & feature engineering
│       ├── modeling.py           # ML model training & evaluation
│       ├── signal.py             # Signal detection algorithms
│       ├── explain.py            # SHAP interpretability
│       ├── descriptive.py        # Descriptive statistics
│       └── utils.py              # Common utilities
├── scripts/                       # Analysis scripts
│   ├── paper_pipeline.py         # Complete 7-experiment pipeline ⭐
│   ├── generate_figures.py       # Publication-quality figures
│   ├── generate_supplementary.py # Supplementary materials
│   └── 01-12_*.py                # Individual analysis scripts
└── outputs/                       # Output directory (created at runtime)
    ├── paper_results/
    │   ├── figures/              # PNG/PDF figures
    │   ├── tables/               # CSV result tables
    │   ├── models/               # Saved model files
    │   ├── reports/              # Paper manuscripts
    │   └── supplementary/        # Supplementary materials
    └── data/                     # Data directory (created at runtime)
        ├── raw/                  # Raw FAERS downloads
        ├── interim/              # Extracted quarterly data
        └── processed/            # Final Parquet datasets
```

## 🚀 Quick Start

### Environment Requirements

- **Python**: 3.10 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Disk Space**: ~10GB for raw data and processed files
- **Operating System**: Windows, macOS, or Linux

### Installation

```bash
# Clone or download this repository
cd faers-sglt2-dka

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install as package
pip install -e .
```

### Data Preparation

1. **Download FAERS Data** from the FDA website:
   - URL: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
   - Download ASCII data files from **2015Q1 to 2025Q4** (ZIP format)
   - Extract ZIP files to `data/raw/` directory

2. **File naming convention**:
   ```
   data/raw/
   ├── 2015Q1aasr.zip
   ├── 2015Q1aexr.zip
   ├── 2015Q1adem.zip
   ├── ...
   └── 2025Q4aasr.zip
   ```

### Running the Analysis

#### Method 1: Complete Pipeline (Recommended)

```bash
# Run the full 7-experiment pipeline (~2-3 minutes)
python scripts/paper_pipeline.py
```

This single script automatically:
1. Extracts raw data to Parquet format
2. Builds analysis datasets with feature engineering
3. Performs signal detection (ROR/PRR)
4. Trains ML models (LightGBM, XGBoost, Ensemble)
5. Conducts SHAP explainability analysis
6. Runs temporal validation and ablation studies
7. Generates all figures and result tables

#### Method 2: Modular Scripts (Step-by-Step)

```bash
# 1. Download and extract FAERS data
python scripts/01_download_faers.py
python scripts/02_extract_to_parquet.py

# 2. Build datasets
python scripts/03_build_datasets.py \
    --config config/terms.yml \
    --interim-dir data/interim \
    --processed-dir data/processed

# 3. Run individual analyses
python scripts/04_signal_detection.py
python scripts/05_train_models.py
python scripts/06_shap_analysis.py
python scripts/07_descriptive_stats.py
python scripts/08_ablation_study.py
python scripts/09_temporal_validation.py
python scripts/10_threshold_analysis.py
python scripts/11_sensitivity_analysis.py
python scripts/12_sensitivity_viz.py

# 4. Generate figures and supplementary materials
python scripts/generate_figures.py
python scripts/generate_supplementary.py
```

## 📊 Output Results

After running the pipeline, results will be available in:

```
outputs/paper_results/
├── figures/
│   ├── paper_figures.png          # 6-panel figure for paper
│   ├── shap_summary.png           # SHAP summary plot
│   ├── calibration_curve.png      # Model calibration
│   ├── sensitivity_comparison.png # Sensitivity analysis
│   └── ...                        # All publication figures
├── tables/
│   ├── descriptive_stats.csv      # Descriptive statistics
│   ├── signal_detection.csv       # ROR/PRR results
│   ├── model_performance.csv      # ML model metrics
│   ├── shap_importance.csv        # SHAP feature rankings
│   ├── ablation_study.csv         # Feature ablation results
│   ├── temporal_validation.csv    # Temporal validation
│   ├── threshold_analysis.csv     # Threshold analysis
│   └── outcome_inclusive_comparison.csv # Sensitivity analysis
├── models/
│   └── best_ensemble.joblib       # Trained ensemble model (3.7 MB)
├── reports/
│   └── CDI_SGLT2_DKA_submission_ready_v4.docx  # Final paper manuscript
└── supplementary/
    └── (additional analysis materials)
```

## 🔬 Key Results

### Signal Detection

| Drug | DKA Reports | ROR | 95% CI | PRR | Signal |
|------|-------------|-----|--------|-----|--------|
| Canagliflozin | 4,617 | 108.28 | 104.69–111.99 | 92.30 | ✅ |
| Empagliflozin | 8,356 | 101.70 | 99.03–104.45 | 89.36 | ✅ |
| Dapagliflozin | 4,855 | 54.76 | 53.04–56.54 | 50.42 | ✅ |
| Ertugliflozin | 112 | 50.95 | 41.95–61.88 | 46.52 | ✅ |

**7 out of 8 SGLT2 inhibitors show significant positive signals.**

### Machine Learning Performance

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|---------|--------|----|-----------|--------|
| LightGBM | 0.912 | 0.602 | 0.527 | 0.412 | 0.845 |
| XGBoost | 0.906 | 0.574 | 0.501 | 0.398 | 0.831 |
| **Ensemble** | **0.912** | **0.605** | **0.531** | **0.421** | **0.852** |

**5-fold CV: ROC-AUC = 0.911 ± 0.003, PR-AUC = 0.608 ± 0.007**

### Temporal Validation

| Dataset | PR-AUC | ROC-AUC |
|---------|--------|---------|
| Training (≤2021) | 0.798 | 0.965 |
| Testing (≥2022) | 0.685 | 0.970 |

**Strong generalization across time periods despite changing reporting patterns.**

### Top SHAP Features

1. Reporter occupation code (occupation)
2. Drug count (concomitant medications)
3. Diabetes indication
4. Report year
5. Event occurrence country

## ⚙️ Configuration

### config/terms.yml

The configuration file defines:

- **DKA case definition**: MedDRA Preferred Terms for DKA identification
- **Study drugs**: List of SGLT2 inhibitors to analyze
- **Drug aliases**: Brand names and alternative spellings

You can customize these terms if needed for your specific use case.

## 🛠️ Methodology

### Data Processing

- **Deduplication**: Based on `caseid`, keeping the latest `fda_dt` version
- **Age handling**: Unified to years, outliers removed (>120 years)
- **Outcome variable exclusion**: Post-event outcomes (death, hospitalization, etc.) excluded from main model to avoid label leakage

### Feature Engineering

- **Raw features**: 40 features (demographics, drugs, indications, reporting)
- **Engineered features**: 21 features (drug risk scores, age strata, interactions)
- **Final features**: 61 features selected via ANOVA F-test

### Model Training

- **Train/test split**: 80%/20% stratified sampling
- **Cross-validation**: 5-fold stratified CV
- **Primary metric**: PR-AUC (Precision-Recall AUC) for imbalanced classification
- **Secondary metrics**: ROC-AUC, F1-score, precision, recall

## ⚠️ Important Notes

1. **Data Size**: FAERS raw data ~2.5GB, processed Parquet files ~500MB
2. **Runtime**: Complete pipeline ~2-3 minutes
3. **Memory**: 8GB RAM minimum (16GB recommended for optimal performance)
4. **Outcome Variables**: Excluded from main model to prevent label leakage - see Discussion section for details
5. **Model Purpose**: Report-level triage tool, NOT patient-level risk prediction
6. **FAERS Limitations**: Spontaneous reporting database without denominator - results reflect reporting patterns, not true incidence

## 📚 Citation

If you use this code for your research, please cite:

```bibtex
@article{faers_sglt2_dka_2026,
  author    = {Research Team},
  title     = {Updated Pharmacovigilance Signal Detection and Explainable Machine-Learning Identification of {SGLT2} Inhibitor-Associated Diabetic Ketoacidosis in {FAERS}},
  journal   = {Clinical Drug Investigation},
  year      = {2026},
  doi       = {10.1007/s40264-026-01630-x},
  url       = {https://github.com/huaibing-xtu/faers-sglt2-dka}
}
```

## 📄 License

This project is released under the **MIT License**.

This software is provided for academic research purposes only.

## 📧 Contact & Issues

- **GitHub Issues**: For bugs, questions, or feature requests
- **Documentation**: See `scripts/README.md` for detailed script usage
- **Chinese Documentation**: See `README-cn.md` for Chinese version
- **Email**: 202521623377@smail.xtu.edu.cn

## 🌟 Star This Project

If you find this project useful for your research, please consider **starring** this repository! ⭐

This helps the research community discover these tools and promotes reproducible pharmacovigilance research.

## 🙏 Acknowledgments

- **FAERS Database**: U.S. Food and Drug Administration (FDA)
- **MedDRA**: Medical Dictionary for Regulatory Activities
- **Dependencies**: pandas, scikit-learn, LightGBM, XGBoost, SHAP, matplotlib, seaborn

---

*Last updated: 2026-06-09*  
*Version: 2.0.0*

*This project is part of the [Clinical Drug Investigation](https://link.springer.com/journal/40264) research on SGLT2 inhibitor safety monitoring.*
