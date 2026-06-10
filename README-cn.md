# FAERS SGLT2-DKA Analysis

基于FAERS数据库的SGLT2抑制剂相关糖尿病酮症酸中毒（DKA）信号挖掘与可解释机器学习分析。

## 📋 项目概述

本项目利用FDA不良事件报告系统（FAERS）数据，对SGLT2抑制剂相关的DKA进行：
1. **信号检测**：使用ROR/PRR进行不成比例分析
2. **机器学习建模**：构建LightGBM/XGBoost集成模型识别DKA报告
3. **可解释性分析**：使用SHAP解释模型决策过程
4. **时序验证**：验证模型在时间维度上的泛化能力

## 📁 项目结构

```
faers_python/
├── src/                         # 源代码模块
│   └── faers_sglt2_dka/        # 核心功能模块
│       ├── __init__.py         # 包初始化
│       ├── download.py         # 数据下载模块
│       ├── io.py               # 数据读写模块
│       ├── preprocess.py       # 数据预处理模块
│       ├── modeling.py         # 模型训练模块
│       ├── signal.py           # 信号检测模块
│       ├── explain.py          # SHAP解释模块
│       ├── descriptive.py      # 描述性统计模块
│       └── utils.py            # 工具函数模块
├── scripts/                     # 分析脚本
│   ├── paper_pipeline.py       # 论文完整流水线（推荐）
│   ├── generate_figures.py     # 生成图表
│   ├── generate_supplementary.py  # 生成补充材料
│   └── 01-10_*.py             # 模块化分析脚本
├── config/                      # 配置文件
│   └── terms.yml               # MedDRA术语配置
└── README.md                    # 本文件
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- pip 或 conda

### 安装依赖

```bash
# 使用conda创建虚拟环境
conda create -n faers python=3.10
conda activate faers

# 安装依赖
pip install pandas numpy scikit-learn lightgbm xgboost shap matplotlib seaborn pyarrow pyyaml joblib
```

### 数据准备

1. 从FDA官网下载FAERS季度数据文件：
   - 网址：https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
   - 下载2015Q1-2025Q4的ASCII数据文件（ZIP格式）

2. 将下载的ZIP文件放入 `data/raw/` 目录

### 运行分析

#### 方法一：运行完整流水线（推荐）

```bash
# 运行论文完整分析流水线
python scripts/paper_pipeline.py
```

该脚本将自动执行以下步骤：
1. 解压数据为Parquet格式
2. 构建数据集和特征工程
3. 进行信号检测（ROR/PRR）
4. 训练机器学习模型
5. 进行SHAP可解释性分析
6. 时序验证和特征消融实验
7. 生成所有图表和结果文件

#### 方法二：运行模块化脚本

```bash
# 1. 下载数据
python scripts/01_download_faers.py

# 2. 解压为Parquet格式
python scripts/02_extract_to_parquet.py

# 3. 构建数据集
python scripts/03_build_datasets.py --config config/terms.yml --interim-dir data/interim --processed-dir data/processed

# 4. 信号检测
python scripts/04_signal_detection.py

# 5. 模型训练
python scripts/05_train_models.py

# 6. SHAP分析
python scripts/06_shap_analysis.py

# 7. 描述性统计
python scripts/07_descriptive_stats.py

# 8. 特征消融实验
python scripts/08_ablation_study.py

# 9. 时序验证
python scripts/09_temporal_validation.py

# 10. 阈值敏感性分析
python scripts/10_threshold_analysis.py
```

#### 生成图表

```bash
# 生成论文图表（Times New Roman字体，300 DPI）
python scripts/generate_figures.py

# 生成补充材料
python scripts/generate_supplementary.py
```

## 📊 输出结果

运行完成后，结果将保存在以下目录：

```
outputs/
├── paper_results/
│   ├── figures/           # 图表（PNG和PDF格式）
│   ├── tables/            # 表格数据（CSV格式）
│   ├── reports/           # 论文稿件
│   ├── models/            # 训练好的模型
│   └── supplementary/     # 补充材料
└── data/
    ├── interim/           # 中间数据（Parquet格式）
    └── processed/         # 处理后的数据
```

## 🔧 配置说明

### config/terms.yml

MedDRA术语配置文件，定义DKA相关术语：

```yaml
target_event_terms:
  core:
    - Diabetic ketoacidosis
    - Ketoacidosis
    - Euglycaemic diabetic ketoacidosis
    - Diabetic ketosis
    - Ketosis
    - Acidosis
    - Metabolic acidosis
    - Blood ketone body increased
```

## 📈 主要结果

### 信号检测
- 7/8 SGLT2抑制剂显示显著阳性信号
- 卡格列净信号最强（ROR = 108.28, 95% CI: 104.69–111.99）

### 机器学习模型
| 模型 | PR-AUC | ROC-AUC | F1 |
|------|--------|---------|-----|
| LightGBM | 0.602 | 0.912 | 0.527 |
| XGBoost | 0.574 | 0.906 | 0.501 |
| Ensemble | 0.605 | 0.912 | 0.531 |

### SHAP分析 Top 5特征
1. 报告者职业代码（OccCod）
2. 药物数量（DrgCnt）
3. 糖尿病适应症（IndDM）
4. 报告年份（RptYr）
5. 事件发生国家（OccCtry）

## 📝 方法说明

### 数据处理
- **去重策略**：基于caseid去重，保留最新fda_dt版本
- **年龄变量**：统一换算为年，移除异常值（>120岁）
- **结局变量排除**：为避免标签泄漏，排除hospitalization、death等结局变量

### 特征工程
- **原始特征**：40个（人口学、药物、适应症、报告特征）
- **工程化特征**：21个（药物风险评分、年龄分层、交互项等）
- **最终特征**：61个（经ANOVA F-test筛选）

### 模型训练
- **训练集/测试集划分**：80%/20%，分层抽样
- **交叉验证**：5折分层交叉验证
- **评估指标**：PR-AUC（主要）、ROC-AUC、F1

## ⚠️ 注意事项

1. **数据大小**：FAERS原始数据约2.5GB，处理后的Parquet文件约500MB
2. **运行时间**：完整流水线约需2-3分钟
3. **内存需求**：建议至少8GB RAM
4. **结局变量**：本项目排除了结局变量（hospitalization, death等）以避免标签泄漏

## 📚 参考文献

1. Kitabchi AE, et al. Hyperglycemic crises in adult patients with diabetes. Diabetes Care. 2009;32(7):1335-1343.
2. Peters AL, et al. Euglycemic diabetic ketoacidosis: a potential complication of treatment with SGLT2 inhibition. Diabetes Care. 2015;38(9):1687-1693.
3. Fadini GP, et al. SGLT2 inhibitors and diabetic ketoacidosis: data from the FDA Adverse Event Reporting System. Diabetologia. 2017;60(8):1385-1389.

## 📄 License

本项目仅用于学术研究目的。

## 📧 Contact

如有问题，请通过GitHub Issues联系。

---

*最后更新：2026-06-04*
