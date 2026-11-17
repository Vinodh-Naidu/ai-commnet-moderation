
MODEL_NAME = "distilbert/distilbert-base-uncased"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

NUM_LABELS = len(LABELS)

MAX_LENGTH = 256

BATCH_SIZE = 16

EPOCHS = 3

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

RANDOM_SEED = 42

TRAIN_FILE = "data/processed/train_processed.csv"

VALIDATION_FILE = "data/processed/validation_processed.csv"

MODEL_OUTPUT_DIR = "model/distilbert-toxic-classifier"
