"""Task 3: classic machine learning (Logistic Regression and Decision Tree).

Run:  python src/task3_ml_models.py
"""
from sklearn.metrics import ConfusionMatrixDisplay

from common import (
    decision_tree,
    evaluate,
    fmt_table,
    load_data,
    logistic_regression,
    pd,
    plt,
    save_fig,
    split_data,
)
from config import RESULTS_DIR


def main() -> pd.DataFrame:
    print("Task 3 - machine learning classification")
    X_train, X_test, y_train, y_test = split_data(load_data())
    print(f"  train: {len(X_train):,}  test: {len(X_test):,}")

    models = {
        "Logistic Regression": logistic_regression(),
        "Decision Tree (depth 6)": decision_tree(),
    }
    rows = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, pipe) in zip(axes, models.items()):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows[name] = evaluate(y_test, pred)
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["normal", "suspicious"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name)
        ax.grid(False)
    save_fig(fig, "fig4_confusion_matrices.png")

    table = pd.DataFrame(rows).T
    table = table.rename(columns={"fn": "missed_attacks_fn", "fp": "legit_flagged_fp"})
    table.to_csv(RESULTS_DIR / "task3_ml_metrics.csv")
    print(fmt_table(table[["accuracy", "precision", "recall", "f1",
                           "missed_attacks_fn", "legit_flagged_fp"]]))
    return table


if __name__ == "__main__":
    main()
