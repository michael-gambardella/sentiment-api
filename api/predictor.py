"""Inference engine: loads the fine-tuned model once and serves sentiment predictions."""
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from api.errors import ModelNotLoadedError, PredictionError
from config import settings
from data.pipeline import MODEL_NAME, MAX_LENGTH

logger = structlog.get_logger(module=__name__)

ARTIFACTS_DIR = settings.artifacts_dir
LABELS = ["NEGATIVE", "POSITIVE"]


class Predictor:
    """Loads the fine-tuned model once and serves inference requests.

    Intended to be instantiated once at API startup via FastAPI's lifespan,
    then reused for every request. Loading per-request would add ~2s latency.
    """

    def __init__(self, artifact_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not artifact_dir.exists():
            raise ModelNotLoadedError(
                f"Model artifacts not found at '{artifact_dir}'. Run 'make train' first."
            )

        logger.info("Loading tokenizer", model_name=MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        logger.info("Loading model", artifact_dir=str(artifact_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
        self.model.to(self.device)
        self.model.eval()  # disable dropout permanently for inference

        logger.info("Predictor ready", device=str(self.device))

    def predict(self, text: str) -> dict:
        """Run inference on a single text input.

        Args:
            text: Raw input string.

        Returns:
            Dict with 'label' (str) and 'confidence' (float) keys.

        Raises:
            PredictionError: If tokenisation or model inference fails unexpectedly.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():  # no gradient graph needed during inference
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # softmax converts raw logits to probabilities that sum to 1
            probs = F.softmax(outputs.logits, dim=-1)
            predicted_idx = probs.argmax(dim=-1).item()
            confidence = probs[0, predicted_idx].item()

            return {
                "label": LABELS[predicted_idx],
                "confidence": round(confidence, 4),
            }
        except (ModelNotLoadedError, PredictionError):
            raise
        except Exception as exc:
            raise PredictionError(f"Inference failed: {exc}") from exc
