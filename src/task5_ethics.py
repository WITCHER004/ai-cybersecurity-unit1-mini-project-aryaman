"""Task 5: ethical experiments.

  5a. Threshold trade-off: blocked legitimate users vs missed attacks.
  5b. Bias simulation: over-flagged VPN / international users in the training labels.
  5c. Explainability: permutation importance of the neural network.

Run:  python src/task5_ethics.py
"""
from sklearn.inspection import permutation_importance

from common import (
    evaluate,
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
from config import FEATURES, RESULTS_DIR, SEED

BIAS_FLIP_SHARE = 0.35


def threshold_experiment(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    print("  5a. threshold trade-off (Logistic Regression)")
    pipe = logistic_regression().fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows = []
    for t in np.round(np.arange(0.1, 1.0, 0.1), 1):
        m = evaluate(y_test, (proba >= t).astype(int))
        rows.append({"threshold": t, **m})
    table = pd.DataFrame(rows).set_index("threshold")
    table.to_csv(RESULTS_DIR / "task5_threshold_tradeoff.csv")
    print(fmt_table(table[["precision", "recall", "fp", "fn"]]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(table.index, table["fp"], "o-", color="#c44e52", label="legit users blocked (FP)")
    axes[0].plot(table.index, table["fn"], "s-", color="#4c72b0", label="attacks missed (FN)")
    axes[0].set_xlabel("decision threshold")
    axes[0].set_ylabel("number of events (test set)")
    axes[0].set_title("Threshold trade-off")
    axes[0].legend()
    axes[1].plot(table["recall"], table["precision"], "o-", color="#55a868")
    for t, r in table.iterrows():
        axes[1].annotate(f"{t:.1f}", (r["recall"], r["precision"]), fontsize=8,
                         xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision vs recall (labels = threshold)")
    save_fig(fig, "fig6_threshold_tradeoff.png")
    return table


def group_masks(X: pd.DataFrame) -> dict:
    return {
        "domestic, no VPN": (X["country_group"] == "domestic") & (X["is_vpn"] == 0),
        "international": X["country_group"] == "international",
        "VPN user": X["is_vpn"] == 1,
    }


def false_positive_rates(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    legit = y_test.to_numpy() == 0
    return {g: 100 * pred[legit & m.to_numpy()].mean() for g, m in group_masks(X_test).items()}


def coefficient(pipe, feature: str) -> float:
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_[0]
    return float(coefs[list(names).index(feature)])


def bias_experiment(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    print("  5b. bias simulation")
    rng = np.random.default_rng(SEED)
    target_group = (y_train == 0) & ((X_train["is_vpn"] == 1) |
                                     (X_train["country_group"] == "international"))
    candidates = np.flatnonzero(target_group.to_numpy())
    flip = rng.choice(candidates, int(BIAS_FLIP_SHARE * len(candidates)), replace=False)
    y_biased = y_train.copy()
    y_biased.iloc[flip] = 1
    print(f"     relabelled {len(flip):,} legitimate VPN/international events as suspicious; "
          f"suspicious share {y_train.mean():.1%} -> {y_biased.mean():.1%}")

    fair = logistic_regression().fit(X_train, y_train)
    biased = logistic_regression().fit(X_train, y_biased)

    fpr = pd.DataFrame({
        "correct labels": false_positive_rates(fair, X_test, y_test),
        "biased labels": false_positive_rates(biased, X_test, y_test),
    })
    fpr.to_csv(RESULTS_DIR / "task5_bias_fpr_by_group.csv")
    print("     false positive rate (%) among legitimate users, evaluated on correct labels:")
    print(fmt_table(fpr))

    overall = pd.DataFrame({
        "correct labels": evaluate(y_test, fair.predict(X_test)),
        "biased labels": evaluate(y_test, biased.predict(X_test)),
    }).T
    overall["coef_is_vpn"] = [coefficient(fair, "num__is_vpn"), coefficient(biased, "num__is_vpn")]
    overall["coef_international"] = [
        coefficient(fair, "cat__country_group_international"),
        coefficient(biased, "cat__country_group_international"),
    ]
    overall.to_csv(RESULTS_DIR / "task5_bias_overall.csv")
    print("     headline metrics hardly move, coefficients do:")
    print(fmt_table(overall[["accuracy", "f1", "coef_is_vpn", "coef_international"]]))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(fpr))
    ax.bar(x - 0.2, fpr["correct labels"], 0.4, label="trained on correct labels", color="#4c72b0")
    ax.bar(x + 0.2, fpr["biased labels"], 0.4, label="trained on biased labels", color="#c44e52")
    ax.set_xticks(x, fpr.index)
    ax.set_ylabel("false positive rate (%) among legitimate users")
    ax.set_title("Bias simulation: who gets blocked by mistake?")
    ax.legend()
    save_fig(fig, "fig7_bias_simulation.png")
    return fpr


def explainability(X_train, X_test, y_train, y_test) -> pd.Series:
    print("  5c. permutation importance of the neural network")
    pipe = neural_network().fit(X_train, y_train)
    result = permutation_importance(pipe, X_test[FEATURES], y_test, scoring="f1",
                                    n_repeats=10, random_state=SEED)
    imp = pd.Series(result.importances_mean, index=FEATURES).sort_values()
    imp.rename("f1_drop").to_csv(RESULTS_DIR / "task5_permutation_importance.csv")
    print(fmt_table(imp.sort_values(ascending=False).to_frame("F1 drop")))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp.index, imp.values, color="#8172b3")
    ax.set_xlabel("drop in F1 when the feature is shuffled")
    ax.set_title("MLP: permutation importance (post-hoc explanation)")
    save_fig(fig, "fig8_permutation_importance.png")
    return imp


def main() -> None:
    print("Task 5 - ethical experiments")
    data = split_data(load_data())
    threshold_experiment(*data)
    bias_experiment(*data)
    explainability(*data)


if __name__ == "__main__":
    main()
