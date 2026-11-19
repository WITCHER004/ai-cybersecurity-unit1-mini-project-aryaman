"""Task 2: threat landscape analysis (EDA).

Outputs figures 1-3 and CSV tables of class means and correlations.
Run:  python src/task2_eda.py
"""
from common import fmt_table, load_data, pd, plt, save_fig, sns
from config import NUMERIC_FEATURES, RESULTS_DIR, TARGET

PALETTE = {"normal": "#4c72b0", "suspicious": "#c44e52"}


def main() -> None:
    print("Task 2 - threat landscape analysis")
    df = load_data()
    df["class"] = df[TARGET].map({0: "normal", 1: "suspicious"})

    # Figure 1: class balance and attack types.
    counts = df["class"].value_counts().reindex(["normal", "suspicious"])
    attacks = df.loc[df[TARGET] == 1, "attack_type"].value_counts().sort_values()
    print(f"  events: {len(df):,}  suspicious: {counts['suspicious']:,} ({df[TARGET].mean():.1%})")
    print(f"  accuracy of 'everything is normal': {1 - df[TARGET].mean():.1%}")
    print("  attack types:", attacks.sort_values(ascending=False).to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v}", ha="center", va="bottom")
    axes[0].set_title("Normal vs suspicious events")
    axes[0].set_ylabel("count")
    axes[1].barh(attacks.index, attacks.values, color=PALETTE["suspicious"])
    axes[1].set_title("Frequency of suspicious activity types")
    axes[1].set_xlabel("count")
    save_fig(fig, "fig1_class_balance.png")

    # Threat indicators: class means and correlation with the label.
    means = df.groupby("class")[NUMERIC_FEATURES].mean().T
    means["ratio"] = means["suspicious"] / means["normal"]
    corr = df[NUMERIC_FEATURES].corrwith(df[TARGET]).sort_values(key=abs, ascending=False)
    means.to_csv(RESULTS_DIR / "task2_feature_means.csv")
    corr.rename("corr_with_label").to_csv(RESULTS_DIR / "task2_correlations.csv")
    print("  feature means by class:\n" + fmt_table(means))
    print("  correlation with label:\n" + fmt_table(corr.to_frame("corr")))

    # Figure 2: distributions of attack-related features.
    show = ["hour_of_day", "failed_attempts", "requests_per_min",
            "bytes_out_mb", "ip_reputation", "geo_distance_km"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    discrete = {"hour_of_day", "failed_attempts"}
    for ax, col in zip(axes.ravel(), show):
        bins = {"discrete": True} if col in discrete else {"bins": 40}
        sns.histplot(data=df, x=col, hue="class", stat="density", common_norm=False,
                     element="step", palette=PALETTE, ax=ax, **bins)
        ax.set_title(col)
    fig.suptitle("Distribution of attack-related features (normal vs suspicious)")
    save_fig(fig, "fig2_feature_distributions.png")

    # Figure 3: % suspicious per group of each indicator.
    indicators = ["country_group", "device_type", "is_vpn", "new_device"]
    rates = {}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col in zip(axes, indicators):
        rate = df.groupby(col)[TARGET].mean().mul(100)
        rates[col] = rate
        ax.bar(rate.index.astype(str), rate.values, color="#dd8452")
        ax.set_title(f"% suspicious by {col}")
        ax.set_ylabel("%")
    save_fig(fig, "fig3_suspicious_rate_by_indicator.png")
    rate_table = pd.concat(rates, names=["indicator", "value"]).rename("pct_suspicious")
    rate_table.to_csv(RESULTS_DIR / "task2_suspicious_rate_by_group.csv")
    print("  % suspicious by group:\n" + fmt_table(rate_table.to_frame()))


if __name__ == "__main__":
    main()
