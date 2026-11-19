"""Task 4: deep learning prototype (MLP 64-32) and comparison with the ML models.

Run:  python src/task4_deep_learning.py
"""
import time

from common import (
    decision_tree,
    evaluate,
    fit_timed,
    fmt_table,
    load_data,
    logistic_regression,
    neural_network,
    np,
    pd,
    plt,
    save_fig,
    split_data,
)
from config import RESULTS_DIR


def main() -> pd.DataFrame:
    print("Task 4 - deep learning prototype and comparison")
    X_train, X_test, y_train, y_test = split_data(load_data())

    models = {
        "Logistic Regression": logistic_regression(),
        "Decision Tree": decision_tree(),
        "Neural Network (MLP 64-32)": neural_network(),
    }
    rows = {}
    for name, pipe in models.items():
        train_s = fit_timed(pipe, X_train, y_train)
        start = time.perf_counter()
        pred = pipe.predict(X_test)
        predict_s = time.perf_counter() - start
        rows[name] = {**evaluate(y_test, pred), "train_s": train_s, "predict_s": predict_s}

    mlp = models["Neural Network (MLP 64-32)"].named_steps["model"]
    n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    print(f"  MLP parameters: {n_params:,}  epochs before early stopping: {mlp.n_iter_}")

    table = pd.DataFrame(rows).T
    table.to_csv(RESULTS_DIR / "task4_ml_vs_dl.csv")
    print(fmt_table(table[["accuracy", "precision", "recall", "f1", "train_s", "predict_s"]]))

    # Figure 5: performance, training time, MLP training curve.
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(table))
    width = 0.2
    for i, m in enumerate(metrics):
        axes[0].bar(x + (i - 1.5) * width, table[m], width, label=m)
    axes[0].set_xticks(x, table.index, rotation=15, ha="right")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("Detection performance")
    axes[0].legend(fontsize=8)

    axes[1].bar(table.index, table["train_s"], color="#dd8452")
    axes[1].set_xticks(x, table.index, rotation=15, ha="right")
    axes[1].set_title("Training time (seconds)")

    axes[2].plot(mlp.loss_curve_, label="training loss")
    axes[2].plot(mlp.validation_scores_, label="validation accuracy")
    axes[2].set_xlabel("epoch")
    axes[2].set_title("MLP training curve")
    axes[2].legend()
    save_fig(fig, "fig5_ml_vs_dl.png")

    best_ml_f1 = table.loc[["Logistic Regression", "Decision Tree"], "f1"].max()
    gain = table.loc["Neural Network (MLP 64-32)", "f1"] - best_ml_f1
    ratio = table.loc["Neural Network (MLP 64-32)", "train_s"] / table.loc["Logistic Regression", "train_s"]
    print(f"  MLP F1 gain over best classic model: {gain:+.3f}; "
          f"training ~{ratio:.0f}x slower than Logistic Regression")
    return table


if __name__ == "__main__":
    main()
