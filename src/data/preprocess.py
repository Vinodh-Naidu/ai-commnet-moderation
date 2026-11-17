
from pathlib import Path
import html
import re
import sys

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = RAW_DIR / "train.csv"

TRAIN_OUTPUT = PROCESSED_DIR / "train_processed.csv"
VALIDATION_OUTPUT = PROCESSED_DIR / "validation_processed.csv"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Perform conservative text normalization.

    Preserve punctuation, capitalization and word boundaries
    because they contain useful moderation signals.
    """

    text = html.unescape(str(text))

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 3. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:

    print("\n[1/7] Loading training dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset shape: {df.shape}")

    required_columns = ["id", "comment_text"] + LABELS

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values in required columns."
        )

    # Validate binary labels
    for label in LABELS:
        if not set(df[label].unique()).issubset({0, 1}):
            raise ValueError(
                f"Unexpected values found in label: {label}"
            )

    print("Dataset validation successful.")

    return df


# ============================================================
# 4. CLEAN COMMENTS
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[2/7] Cleaning comment text...")

    df = df.copy()

    df["comment_text"] = df["comment_text"].apply(clean_text)

    # Remove empty comments
    original_count = len(df)

    df = df[df["comment_text"].str.len() > 0].copy()

    removed_count = original_count - len(df)

    print(f"Empty comments removed: {removed_count}")
    print(f"Remaining comments: {len(df)}")

    return df


# ============================================================
# 5. MULTI-LABEL STRATIFIED SPLITTING
# ============================================================

def split_dataset(df: pd.DataFrame):

    print("\n[3/7] Preparing feature and label matrices...")

    X = df[["id", "comment_text"]].to_numpy()

    y = df[LABELS].to_numpy(dtype="int32")

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)

    print("\n[4/7] Performing multi-label stratified split...")

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, val_indices = next(
        splitter.split(X, y)
    )

    X_train = X[train_indices]
    X_val = X[val_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))

    # Reconstruct DataFrames
    train_df = pd.DataFrame(
        X_train,
        columns=["id", "comment_text"]
    )

    val_df = pd.DataFrame(
        X_val,
        columns=["id", "comment_text"]
    )

    for index, label in enumerate(LABELS):
        train_df[label] = y_train[:, index]
        val_df[label] = y_val[:, index]

    return train_df, val_df


# ============================================================
# 6. SAVE PROCESSED DATASETS
# ============================================================

def save_datasets(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
):

    print("\n[5/7] Saving processed datasets...")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    val_df.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    print("Training file:", TRAIN_OUTPUT)
    print("Validation file:", VALIDATION_OUTPUT)


# ============================================================
# 7. DATASET REPORT AND VERIFICATION
# ============================================================

def generate_report(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
):

    print("\n[6/7] Generating dataset report...")

    print("\n========== DATASET SUMMARY ==========")

    print("Training shape:", train_df.shape)
    print("Validation shape:", val_df.shape)

    print("\n========== TRAIN LABEL DISTRIBUTION ==========")

    train_distribution = (
            train_df[LABELS].mean() * 100
    ).round(3)

    print(train_distribution)

    print("\n========== VALIDATION LABEL DISTRIBUTION ==========")

    val_distribution = (
            val_df[LABELS].mean() * 100
    ).round(3)

    print(val_distribution)

    print("\n========== POSITIVE LABEL COUNTS ==========")

    comparison = pd.DataFrame({
        "train": train_df[LABELS].sum(),
        "validation": val_df[LABELS].sum()
    })

    print(comparison)

    print("\n[7/7] Verifying generated files...")

    for file in [TRAIN_OUTPUT, VALIDATION_OUTPUT]:

        if not file.exists():
            raise RuntimeError(
                f"Output file was not created: {file}"
            )

        print(
            f"{file.name}: "
            f"{file.stat().st_size:,} bytes"
        )

    # Verify saved data can be read
    saved_train = pd.read_csv(TRAIN_OUTPUT)
    saved_val = pd.read_csv(VALIDATION_OUTPUT)

    assert len(saved_train) == len(train_df)
    assert len(saved_val) == len(val_df)

    assert list(saved_train.columns) == ["id", "comment_text"] + LABELS
    assert list(saved_val.columns) == ["id", "comment_text"] + LABELS

    print("\nDataset preprocessing completed successfully!")


# ============================================================
# MAIN EXECUTION
# ============================================================

def prepare_dataset():

    print("=" * 60)
    print("AI COMMENT MODERATION - DATA PREPROCESSING")
    print("=" * 60)

    print("\nProject root:", ROOT)

    df = load_dataset()

    df = clean_dataset(df)

    train_df, val_df = split_dataset(df)

    save_datasets(train_df, val_df)

    generate_report(train_df, val_df)


if __name__ == "__main__":

    try:
        prepare_dataset()

    except Exception as exc:

        print(
            f"\nPREPROCESSING FAILED: {exc}",
            file=sys.stderr
        )

        raise
