"""Task 1: confirm the environment works.

Prints library versions and trains a tiny model.
Run:  python src/task1_check_setup.py
"""
import platform

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import SEED, ensure_dirs


def main() -> None:
    print("Task 1 - environment check")
    print(f"  python       {platform.python_version()}")
    for name, mod in [
        ("numpy", np),
        ("pandas", pd),
        ("scikit-learn", sklearn),
        ("matplotlib", matplotlib),
        ("seaborn", sns),
    ]:
        print(f"  {name:<12} {mod.__version__}")

    X, y = make_classification(n_samples=500, n_features=5, random_state=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=SEED)
    acc = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)
    print(f"  tiny test model accuracy: {acc:.3f}")

    ensure_dirs()
    print("  setup OK")


if __name__ == "__main__":
    main()
