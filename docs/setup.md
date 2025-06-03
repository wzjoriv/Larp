## 🔧 Development Setup
### ✅ Create Virtual Environment (optional)

```bash
python -m venv .venv
```

Activate the environment in a new terminal.

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
pip install matplotlib pytest ipykernel scipy build twine
```

## 🚀 Deployment
### 🏗️ Build Package

```bash
py -m build
```

### 📤 Upload to PyPI

```bash
py -m twine upload --repository pypi dist/*
```

## ⚠️ Notes

Tested with Python 3.13. Other versions have not been verified.