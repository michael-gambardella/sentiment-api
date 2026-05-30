"""ETL pipeline: loads the IMDB dataset, cleans raw text, tokenizes with DistilBERT, and returns DataLoaders."""
import re
from typing import Tuple

import structlog
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import settings

logger = structlog.get_logger(module=__name__)

MODEL_NAME = settings.model_name
MAX_LENGTH = settings.max_input_length
BATCH_SIZE = settings.batch_size


def clean_text(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_raw_dataset() -> DatasetDict:
    """Download and cache the IMDB dataset from Hugging Face Hub."""
    logger.info("Loading IMDB dataset")
    dataset = load_dataset("imdb")
    logger.info("Dataset loaded", train_samples=len(dataset["train"]), test_samples=len(dataset["test"]))
    return dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """Clean text and apply tokenizer across all dataset splits.

    Each example is transformed into three tensors:
      - input_ids: token IDs from the tokenizer vocabulary
      - attention_mask: 1 for real tokens, 0 for padding
      - labels: 0 (NEGATIVE) or 1 (POSITIVE)
    """
    def tokenize_batch(batch: dict) -> dict:
        cleaned = [clean_text(text) for text in batch["text"]]
        return tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    logger.info("Tokenizing dataset")
    tokenized = dataset.map(tokenize_batch, batched=True)

    # Rename so PyTorch cross-entropy loss finds the target column automatically
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info("Tokenization complete")
    return tokenized


def build_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """Run the full ETL pipeline and return train and test DataLoaders.

    Args:
        batch_size: Number of examples per batch. Default is 16.

    Returns:
        A tuple of (train_loader, test_loader).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_raw_dataset()
    tokenized = tokenize_dataset(dataset, tokenizer)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,   # randomize order each epoch so the model can't learn sequence
    )
    test_loader = DataLoader(
        tokenized["test"],
        batch_size=batch_size,
        shuffle=False,  # evaluation order doesn't matter; keep it deterministic
    )

    logger.info("DataLoaders ready", train_batches=len(train_loader), test_batches=len(test_loader))
    return train_loader, test_loader
