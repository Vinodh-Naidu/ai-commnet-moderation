
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def validate_dataset():
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    test_labels = pd.read_csv(RAW_DIR / "test_labels.csv")

    print("\n========== DATASET SHAPES ==========")
    print("Training:", train.shape)
    print("Test:", test.shape)
    print("Test labels:", test_labels.shape)

    print("\n========== TRAINING COLUMNS ==========")
    print(train.columns.tolist())

    print("\n========== MISSING VALUES ==========")
    print(train.isnull().sum())

    print("\n========== DUPLICATE COMMENTS ==========")
    print(train["comment_text"].duplicated().sum())

    print("\n========== LABEL DISTRIBUTION ==========")
    print(train[LABEL_COLUMNS].sum())

    print("\n========== LABEL VALIDATION ==========")
    for label in LABEL_COLUMNS:
        print(
            label,
            train[label].value_counts().to_dict()
        )

        assert set(train[label].dropna().unique()).issubset({0, 1}), (
            f"Unexpected values in {label}"
        )

    print("\n========== EMPTY COMMENTS ==========")
    print(train["comment_text"].isna().sum())

    print("\n========== TEST LABEL CHECK ==========")
    print(test_labels[LABEL_COLUMNS].apply(
        lambda col: col.value_counts().to_dict()
    ))

    print("\nDataset validation completed!")


if __name__ == "__main__":
    validate_dataset()
