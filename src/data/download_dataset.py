
from pathlib import Path
import gzip
import shutil

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

REPO_ID = "Heliosoph/Jigsaw-Toxic-Comments"

FILES = [
    "train.csv.gz",
    "test.csv.gz",
    "test_labels.csv.gz",
]


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        print(f"Downloading {filename}...")

        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset"
        )

        output_file = RAW_DIR / filename.replace(".gz", "")

        print(f"Extracting {filename}...")

        with gzip.open(downloaded_path, "rb") as source:
            with output_file.open("wb") as destination:
                shutil.copyfileobj(source, destination)

        print(f"Saved: {output_file}")

    print("\nDataset download completed!")


if __name__ == "__main__":
    download_dataset()
