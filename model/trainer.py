import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from data.pipeline import build_dataloaders, MODEL_NAME

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
NUM_LABELS = 2
LEARNING_RATE = 2e-5   # standard fine-tuning LR from the BERT paper; higher risks catastrophic forgetting
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1     # ramp LR from 0 → peak over the first 10% of steps


def get_device() -> torch.device:
    """Return GPU if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


def build_model(num_labels: int = NUM_LABELS) -> AutoModelForSequenceClassification:
    """Load DistilBERT with a randomly-initialized classification head.

    The transformer body starts from pre-trained weights; only the head
    is new and needs to learn from our data.
    """
    logger.info("Loading pre-trained model: %s", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )
    return model


def run_epoch(
    model: AutoModelForSequenceClassification,
    train_loader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one full pass over the training data.

    Returns:
        Average training loss for this epoch.
    """
    model.train()  # activates dropout layers for regularization
    total_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass — HuggingFace models return loss automatically when labels are passed
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()  # clear gradients after each step, not before, to catch any accidental accumulation

        total_loss += loss.item()

        if step % 200 == 0:
            logger.info(
                "Epoch %d | Step %d/%d | Running Avg Loss: %.4f",
                epoch, step, len(train_loader), total_loss / step,
            )

    return total_loss / len(train_loader)


def train(
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = 16,
) -> None:
    """Fine-tune DistilBERT on IMDB and save checkpoints after each epoch.

    Artifacts are written to model/artifacts/:
      checkpoint_epoch_1/, checkpoint_epoch_2/, checkpoint_epoch_3/  ← per-epoch snapshots
      final/                                                          ← the model the API loads

    Args:
        num_epochs: Number of full passes over the training data.
        learning_rate: Peak learning rate reached after warmup.
        batch_size: Examples per gradient update step.
    """
    device = get_device()
    train_loader, _ = build_dataloaders(batch_size=batch_size)
    model = build_model().to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(
        "Starting training — epochs: %d | total steps: %d | warmup steps: %d",
        num_epochs, total_steps, warmup_steps,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        avg_loss = run_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        logger.info("Epoch %d complete | Avg Loss: %.4f", epoch, avg_loss)

        checkpoint_path = ARTIFACTS_DIR / f"checkpoint_epoch_{epoch}"
        model.save_pretrained(checkpoint_path)
        logger.info("Checkpoint saved → %s", checkpoint_path)

    final_path = ARTIFACTS_DIR / "final"
    model.save_pretrained(final_path)
    logger.info("Final model saved → %s", final_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    train()
