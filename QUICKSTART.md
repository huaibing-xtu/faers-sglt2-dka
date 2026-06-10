# Quick Start Guide

## 5-Minute Setup

### 1. Clone Repository

```bash
git clone https://github.com/huaibing-xtu/faers-sglt2-dka.git
cd faers-sglt2-dka
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python validate_installation.py
```

Expected output:
```
============================================================
FAERS SGLT2-DKA Codebase Validation
============================================================

✅ Python 3.10+
✅ All dependencies installed
✅ All files present
✅ Configuration file is valid

✅ All checks passed! The codebase is ready to use.
```

### 5. Download FAERS Data

1. Go to: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
2. Download quarterly data from 2015Q1 to 2025Q4 (ASCII format)
3. Extract ZIP files to `data/raw/` directory

**Example**:
```bash
mkdir -p data/raw
# Copy extracted files to data/raw/
```

### 6. Run Analysis

```bash
# Complete pipeline (recommended)
python scripts/paper_pipeline.py

# Or run individual steps
python scripts/01_download_faers.py
python scripts/02_extract_to_parquet.py
python scripts/03_build_datasets.py --config config/terms.yml
```

### 7. View Results

Results are saved to `outputs/paper_results/`:

```bash
# View figures
ls outputs/paper_results/figures/

# View tables
ls outputs/paper_results/tables/

# View final paper manuscript
ls outputs/paper_results/reports/
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Memory error

**Solution**: Increase RAM or use smaller dataset for testing

### Issue: FAERS data download fails

**Solution**: Check FDA website accessibility or try manual download

## Next Steps

1. **Read the main README.md** for detailed methodology
2. **Check scripts/README.md** for individual script options
3. **Review paper_polished.md** in outputs for research results
4. **Explore the code** in src/faers_sglt2_dka/

## Citation

If you use this code for research, please cite:

```bibtex
@software{faers_sglt2_dka_2026,
  author       = {Research Team},
  title        = {FAERS SGLT2-DKA Analysis: Signal Detection and Explainable Machine Learning},
  year         = 2026,
  version      = {2.0.0},
  url          = {https://github.com/huaibing-xtu/faers-sglt2-dka}
}
```

## Support

- 📖 **Documentation**: See README.md and CONTRIBUTING.md
- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Questions**: Use GitHub Discussions (if enabled)
- 📧 **Contact**: huaibing@xtu.edu.cn

---

*Happy coding! 🚀*
