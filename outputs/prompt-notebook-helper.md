# Notebook Helper

## 1. Out-of-order execution

Problem:

The notebook works now, but fails when someone runs it from top to bottom.

Reason:

Cells were executed in a different order.

Fix:

Use Kernel > Restart Kernel and Run All Cells before sharing the notebook.

---

## 2. Hidden state

Problem:

A variable still exists even after the cell that created it was deleted.

Reason:

The kernel still keeps old variables in memory.

Fix:

Restart the kernel regularly.

---

## 3. Missing package

Problem:

ImportError or ModuleNotFoundError.

Reason:

The package is not installed in the current notebook environment.

Fix:

Install it with pip in the correct virtual environment.

---

## 4. Plot does not show

Problem:

matplotlib code runs but no chart appears.

Fix:

Use:

%matplotlib inline

and then call:

plt.show()

---

## 5. Notebook uses the wrong Python environment

Problem:

Packages are installed in .venv, but the notebook cannot import them.

Fix:

Choose the correct kernel:

Python (.venv ai-course)
