# Contributing Guide

## Setup

```bash
git clone https://github.com/yourusername/physics_inference.git
cd physics_inference
python -m venv venv_3.12
source venv_3.12/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/run_all_checks.py  # Verify setup
```

## Standards

**Code Quality**:
- 100% MyPy compliance (strict for core tracking, standard for detection)
- PEP 8 style, comprehensive docstrings
- 100% test coverage for new features

**Workflow**:
```bash
git checkout -b feature/name
python scripts/pre_commit_checks.py
# Make changes
python scripts/run_all_checks.py
```

**Testing**:
- `python scripts/run_tests.py` - All tests
- `python scripts/run_mypy.py` - Type checking  
- `python scripts/run_all_checks.py` - Complete QA

## Architecture

**Modules**: `detection/` (inference), `tracking/` (algorithms), `common/` (utilities), `scripts/` (tools)

**Principles**: Interface-based design, dependency injection, comprehensive error handling

**Contribution Areas**:
- Additional tracker implementations
- Performance optimizations
- Physics-based motion prediction
- Game state recognition

## Pull Requests

**Requirements**:
- All quality checks pass (`python scripts/run_all_checks.py`)
- Tests for new functionality
- Updated documentation and type hints
- Descriptive commit messages

## Issues & Support

**Bug Reports**: Include reproduction steps, environment details, and expected vs actual behavior

**Feature Requests**: Provide clear use case and consider contributing to implementation

**Help**: Check documentation, search existing issues, use GitHub Discussions
