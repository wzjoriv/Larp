# Development Setup

---
## Setup
### Create Virtual environment
***Optional***

```bash
python -m venv .venv
```

Then, open a new terminal and activate environment

### Install dependencies

```bash
pip install -r requirements.txt
```

## Deployment

### Build

```bash
py -m build
```

### Update PiPy
```bash
py -m twine upload --repository pypi dist/*
```

## Notes 

1. The code is tested for Python 3.13. No other version is supported.