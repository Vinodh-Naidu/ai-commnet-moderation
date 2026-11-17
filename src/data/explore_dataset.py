
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Positive and negative label distributions.
# Label co-occurrence.
# comment length statistics.
# Toxicity combinations.
# Distribution of comment lengths.

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


def explore_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "train.csv")

    # Comment length
    df["comment_length"] = df["comment_text"].str.len()
    df["word_count"] = df["comment_text"].str.split().str.len()

    print("\n===== COMMENT STATISTICS =====")
    print(df[["comment_length", "word_count"]].describe())

    # Label distribution
    print("\n===== LABEL DISTRIBUTION (%) =====")
    print((df[LABELS].mean() * 100).round(3))

    # Multi-label statistics
    df["label_count"] = df[LABELS].sum(axis=1)

    print("\n===== LABEL COMBINATIONS =====")
    print(df["label_count"].value_counts().sort_index())

    # Save label distribution
    label_counts = df[LABELS].sum().sort_values(
        ascending=False
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title("Positive Comments per Toxicity Category")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_distribution.png")
    plt.close()

    # Label co-occurrence
    cooccurrence = df[LABELS].T.dot(df[LABELS])

    print("\n===== LABEL CO-OCCURRENCE =====")
    print(cooccurrence)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("Toxicity Label Co-occurrence")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_cooccurrence.png")
    plt.close()

    # Comment length distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(
        df["word_count"].clip(upper=300),
        bins=50
    )
    plt.title("Comment Word Count Distribution")
    plt.xlabel("Word Count (capped at 300)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comment_lengths.png")
    plt.close()

    print("\nEDA completed.")
    print(f"Charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    explore_dataset()
