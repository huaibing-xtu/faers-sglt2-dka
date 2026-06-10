# Contributing to FAERS SGLT2-DKA Analysis

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🎯 Scope

This project focuses on:
- Pharmacovigilance signal detection using FAERS data
- Explainable machine learning for adverse event reporting
- Reproducible research in drug safety

## 🛠️ How to Contribute

### Reporting Bugs

1. **Check existing issues**: Search open and closed issues for similar problems
2. **Create a new issue**: Use the bug report template
3. **Include**:
   - Description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Error traceback (if applicable)

### Suggesting Features

1. **Check existing issues**: Ensure your feature hasn't been suggested
2. **Create a feature request**: Use the enhancement template
3. **Include**:
   - Feature description
   - Use case scenario
   - Expected benefits
   - Alternative solutions considered

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly**: Run the complete pipeline to ensure no regressions
5. **Document changes**: Update README and relevant documentation
6. **Commit**: Use clear, descriptive commit messages
7. **Push and create PR**: `git push origin feature/amazing-feature`
8. **Request review**: Tag maintainers for review

### Code Style Guidelines

- **Python**: Follow PEP 8 style guidelines
- **Type hints**: Use type annotations for function parameters and return values
- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Comments**: Add comments for complex logic and non-obvious operations
- **Imports**: Organize imports alphabetically, standard libs before third-party

## 📋 Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/huaibing-xtu/faers-sglt2-dka.git
cd faers-sglt2-dka/code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # Install development dependencies

# Run tests (if applicable)
pytest tests/  # Future testing framework
```

## 🧪 Testing

While the current project focuses on research code, we encourage:

1. **Manual testing**: Run scripts with sample data
2. **Regression testing**: Verify existing results haven't changed
3. **Documentation testing**: Ensure all examples in README work

## 📝 Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Testing additions/changes
- `chore`: Maintenance tasks

**Examples:**
```
feat: add sensitivity analysis for outcome-inclusive models
fix: correct age unit conversion for month inputs
docs: update README with new installation instructions
refactor: optimize feature engineering pipeline
```

## 🔍 Code Review

All contributions require review. Reviewers will check:

1. **Functionality**: Does the code work as intended?
2. **Code quality**: Is the code clean and maintainable?
3. **Documentation**: Is everything properly documented?
4. **Testing**: Has the code been tested?
5. **Consistency**: Does it follow project conventions?

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for contributing to reproducible pharmacovigilance research!

---

*Last updated: 2026-06-09*
