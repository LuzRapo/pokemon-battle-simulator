# pokemon-battle-simulator
Engine for orchestrating 6v6 Pokemon battles.

[![tests](https://github.com/LuzRapo/pokemon-battle-simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/LuzRapo/pokemon-battle-simulator/actions/workflows/tests.yml)

[![linters](https://github.com/LuzRapo/pokemon-battle-simulator/actions/workflows/linters.yml/badge.svg)](https://github.com/LuzRapo/pokemon-battle-simulator/actions/workflows/linters.yml)

## Features
- None, yet.

## Setup
### Check Requirements:
- [Python 3.12+](https://www.python.org/downloads/release/python-3120/)
- [PyCharm 2024+](https://www.jetbrains.com/pycharm/download/)
- [Git (latest)](https://git-scm.com/downloads)
- [uv (latest)](https://github.com/astral-sh/uv)
- Windows or Linux

### In Powershell (Windows) or terminal (Linux):
```bash
cd <your-dev-folder>
git clone https://github.com/LuzRapo/pokemon-battle-simulator.git
cd pokemon-battle-simulator
uv sync --all-extras
```

### In PyCharm:
- Go to File → Settings → Project → Python Interpreter.
- Click the ⚙️ (gear) → Add Interpreter → Add Local Interpreter.
- Choose Existing environment, then browse to: `<your-dev-folder>\.venv\Scripts\python.exe`
- OK → Apply.
```bash
uv run pre-commit install
```

## Running:
- Manually go to `main.py` and press Run ▶ in PyCharm
- Or, in console:
```bash
uv run python main.py
```

## Developer Info
- We use [ruff](https://github.com/astral-sh/ruff) for linting & formatting.
- We use [uv](https://github.com/astral-sh/uv) for dependency & project management.
- We use [mypy](https://github.com/python/mypy) for static type hinting.
- We use [vulture](https://github.com/jendrikseipp/vulture) for finding & removing dead code.

### Contributing:
1. Create a feature branch:
```bash
git checkout -b feature/my-change
```
2. Do work, then make a commit:
```bash
git add -A
git commit -m "Add cool thing"
```
3. Push the branch (if it's a new branch, use `-u`):
```
git push -u origin feature/my-change
```
4. Open a PR on GitHub and request a review.