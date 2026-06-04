# VS Code Helper

## 1. Recommended editor

VS Code is used as the main editor for AI engineering work.

It supports Python, Jupyter notebooks, Git, terminals, debugging, formatting, and remote development.

---

## 2. Essential extensions

Important extensions:

- Python
- Pylance
- Jupyter
- GitLens
- Remote SSH
- WSL
- Python Debugger
- Black Formatter
- Ruff

---

## 3. Python interpreter

The project should use:

.venv/bin/python

This keeps packages isolated inside the project virtual environment.

---

## 4. Useful settings

Important settings:

- formatOnSave
- typeCheckingMode: basic
- notebook output scrolling
- auto save
- terminal scrollback

---

## 5. Daily workflow

Open the project in WSL mode.

Activate the environment:

source .venv/bin/activate

Run scripts in the VS Code terminal.

Check changes before committing:

git status
