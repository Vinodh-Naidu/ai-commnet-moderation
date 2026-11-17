
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.modeling.config import (
    MODEL_NAME,
    LABELS,
    NUM_LABELS
)


def test_model():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")

    print("\nLoading DistilBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    model.config.id2label = {
        i: label for i, label in enumerate(LABELS)
    }

    model.config.label2id = {
        label: i for i, label in enumerate(LABELS)
    }

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print("Device:", device)

    sample = [
        "You are a terrible person.",
        "Thank you for your help!"
    ]

    inputs = tokenizer(
        sample,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.sigmoid(outputs.logits)

    print("\nOutput shape:", outputs.logits.shape)
    print("Expected shape:", (2, NUM_LABELS))

    print("\nModel initialization successful!")


if __name__ == "__main__":
    test_model()
