"""
Quick validation script to check if the codebase is properly set up.
Run this after cloning the repository to verify installation.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is >= 3.10"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python version {version.major}.{version.minor} is too old. Need 3.10+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required = [
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "joblib",
        "pyarrow",
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - NOT INSTALLED")
            missing.append(pkg)

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    print(f"\n✅ All {len(required)} dependencies installed")
    return True

def check_file_structure():
    """Check if required files exist"""
    root = Path(__file__).resolve().parent

    required_files = [
        "README.md",
        "setup.py",
        "requirements.txt",
        "config/terms.yml",
        "src/faers_sglt2_dka/__init__.py",
        "src/faers_sglt2_dka/preprocess.py",
        "src/faers_sglt2_dka/modeling.py",
        "scripts/paper_pipeline.py",
    ]

    missing = []
    for f in required_files:
        if (root / f).exists():
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} - NOT FOUND")
            missing.append(f)

    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    print(f"\n✅ All {len(required_files)} files present")
    return True

def check_config():
    """Check if configuration is valid"""
    try:
        import yaml
        config_path = Path(__file__).resolve().parent / "config" / "terms.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required_keys = ["target_event_terms", "study_drugs"]
        missing_keys = [k for k in required_keys if k not in config]

        if missing_keys:
            print(f"❌ Missing config keys: {', '.join(missing_keys)}")
            return False

        print("✅ Configuration file is valid")
        print(f"   - Target event terms: {len(config['target_event_terms']['core'])} core terms")
        print(f"   - Study drugs: {len(config['study_drugs'])} drugs")
        return True

    except Exception as e:
        print(f"❌ Config check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("FAERS SGLT2-DKA Codebase Validation")
    print("=" * 60)
    print()

    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("File Structure", check_file_structure()),
        ("Configuration", check_config()),
    ]

    print()
    print("=" * 60)
    all_passed = all(result for _, result in checks)

    if all_passed:
        print("✅ All checks passed! The codebase is ready to use.")
        print("\nNext steps:")
        print("1. Download FAERS data from https://fis.fda.gov")
        print("2. Place data in data/raw/ directory")
        print("3. Run: python scripts/paper_pipeline.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")

    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
