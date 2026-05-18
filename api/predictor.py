import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.pipeline import MODEL_NAME, MAX_LENGTH

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "model" / "artifacts" / "final"
LABELS = ["NEGATIVE", "POSITIVE"]


class Predictor:
    """Loads the fine-tuned model once and serves inference requests.

    Intended to be instantiated once at API startup via FastAPI's lifespan,
    then reused for every request. Loading per-request would add ~2s latency.
    """

    def __init__(self, artifact_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading tokenizer from %s", MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        logger.info("Loading model from %s", artifact_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
        self.model.to(self.device)
        self.model.eval()  # disable dropout permanently for inference

        logger.info("Predictor ready on device: %s", self.device)

    def predict(self, text: str) -> dict:
        """Run inference on a single text input.

        Args:
            text: Raw input string.

        Returns:
            Dict with 'label' (str) and 'confidence' (float) keys.
        """
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
