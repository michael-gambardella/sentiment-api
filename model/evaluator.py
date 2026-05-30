"""Evaluation utilities: accuracy, F1 score, and confusion matrix for the fine-tuned classifier."""
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import structlog
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification

from data.pipeline import build_dataloaders
from model.trainer import ARTIFACTS_DIR, get_device

logger = structlog.get_logger(module=__name__)

LABELS = ["NEGATIVE", "POSITIVE"]


def load_model(artifact_dir: Path = ARTIFACTS_DIR / "final") -> AutoModelForSequenceClassification:
    """Load the fine-tuned model from a saved artifact directory."""
    logger.info("Loading model", artifact_dir=str(artifact_dir))
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
    return model


def collect_predictions(
    model: AutoModelForSequenceClassification,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """Run inference over the full test set.

    Returns:
        A tuple of (predictions, ground_truth_labels) as plain Python lists.
    """
    model.eval()  # disables dropout â€” makes predictions deterministic

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():  # no gradient graph needed; saves memory during inference
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # argmax over the last dim: [batch_size, num_labels] â†’ [batch_size]
            preds = outputs.logits.argmax(dim=-1)

            # move to CPU and convert to list so sklearn can consume them
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute accuracy and binary F1 score."""
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    logger.info("Evaluation complete", accuracy=round(accuracy, 4), f1=round(f1, 4))
    return {"accuracy": accuracy, "f1": f1}


def plot_confusion_matrix(
    preds: List[int],
    labels: List[int],
    output_path: Path = ARTIFACTS_DIR / "confusion_matrix.png",
) -> None:
    """Generate a confusion matrix heatmap and save it to disk.

    Rows = actual class, columns = predicted class.
    Diagonal = correct predictions. Off-diagonal = errors.
    """
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix â€” IMDB Sentiment Classifier")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.info("Confusion matrix saved", path=str(output_path))


def evaluate(batch_size: int = 16) -> Dict[str, float]:
    """Full evaluation pipeline: load model, run inference, log metrics, save confusion matrix."""
    device = get_device()
    _, test_loader = build_dataloaders(batch_size=batch_size)
    model = load_model().to(device)

    preds, labels = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(preds, labels)
    plot_confusion_matrix(preds, labels)

    return metrics


if __name__ == "__main__":
    from config import settings
    from logging_config import configure_logging
    configure_logging(settings.environment, settings.log_level)
    evaluate()
