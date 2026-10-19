"""Step 2: exploratory data analysis. Saves figures to reports/figures/.

Usage:  python scripts/eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from heart import config  # noqa: E402
from heart.data import load_clean  # noqa: E402

sns.set_theme(style="whitegrid", palette="Set2")
OUT = config.FIGURES_DIR
OUT.mkdir(parents=True, exist_ok=True)
LABELS = {0: "No disease", 1: "Disease"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print("saved", OUT / name)


def main():
    df = load_clean()
    tgt = df[config.TARGET].map(LABELS)

    # 1. class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = tgt.value_counts()
    ax.bar(counts.index, counts.values, color=["#66c2a5", "#fc8d62"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("patients")
    save(fig, "01_class_balance.png")

    # 2. missing values
    miss = df.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(miss.index, miss.values, color="#8da0cb")
    ax.set_title("Missing values per column (imputed in pipeline)")
    save(fig, "02_missing_values.png")

    # 3. numeric histograms split by target
    num = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, num):
        sns.histplot(data=df.assign(target=tgt), x=col, hue="target", kde=True,
                     element="step", stat="density", common_norm=False, ax=ax)
        ax.set_title(col)
    axes.flat[-1].axis("off")
    save(fig, "03_numeric_histograms.png")

    # 4. correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Feature correlation (Pearson)")
    save(fig, "04_correlation_heatmap.png")

    # 5. disease rate by categorical feature
    cats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.flat, cats):
        rate = df.groupby(col)[config.TARGET].mean()
        ax.bar(rate.index.astype(str), rate.values, color="#e78ac3")
        ax.set_ylim(0, 1)
        ax.set_title(f"Disease rate by {col}")
    save(fig, "05_disease_rate_by_category.png")

    # 6. boxplots of numeric vs target
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, num):
        sns.boxplot(x=tgt, y=df[col], ax=ax)
        ax.set_xlabel("")
        ax.set_title(col)
    save(fig, "06_boxplots_by_target.png")

    # text summary
    summary = df.describe().T.round(2)
    (OUT / "eda_summary.md").write_text(
        "# EDA summary\n\n"
        f"Rows: {len(df)}  |  Positive rate: {df[config.TARGET].mean():.3f}\n\n"
        f"Missing: {miss.to_dict()}\n\n"
        "Top correlations with target:\n\n"
        + corr[config.TARGET].drop(config.TARGET).sort_values(key=abs, ascending=False)
        .round(3).to_markdown() + "\n\n## describe()\n\n" + summary.to_markdown() + "\n"
    )
    print(corr[config.TARGET].sort_values())


if __name__ == "__main__":
    main()
