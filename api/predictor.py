"""Inference engine: loads the fine-tuned model once and serves sentiment predictions."""
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F
from opentelemetry import trace
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from api.errors import ModelNotLoadedError, PredictionError
from config import settings

logger = structlog.get_logger(module=__name__)
tracer = trace.get_tracer(__name__)

ARTIFACTS_DIR = settings.artifacts_dir
MAX_LENGTH = settings.max_input_length
LABELS = ["NEGATIVE", "POSITIVE"]


class Predictor:
    """Loads the fine-tuned model once and serves inference requests.

    Intended to be instantiated once at API startup via FastAPI's lifespan,
    then reused for every request. Loading per-request would add ~2s latency.
    """

    def __init__(self, artifact_dir: Path = ARTIFACTS_DIR, tokenizer_dir: Path | None = None) -> None:
        self.artifact_dir = artifact_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not artifact_dir.exists():
            raise ModelNotLoadedError(
                f"Model artifacts not found at '{artifact_dir}'. Run 'make train' first."
            )

        # Tokenizer doesn't change between epochs; checkpoints may omit it.
        # Fall back to a shared tokenizer dir (e.g. 'final') when not present.
        tok_dir = tokenizer_dir if tokenizer_dir is not None else artifact_dir
        logger.info("Loading tokenizer", artifact_dir=str(artifact_dir), tokenizer_dir=str(tok_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)

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
        with tracer.start_as_current_span("model.inference") as span:
            span.set_attribute("input.length", len(text))
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

                result = {
                    "label": LABELS[predicted_idx],
                    "confidence": round(confidence, 4),
                }
                span.set_attribute("prediction.label", result["label"])
                span.set_attribute("prediction.confidence", result["confidence"])
                return result
            except (ModelNotLoadedError, PredictionError):
                raise
            except Exception as exc:
                raise PredictionError(f"Inference failed: {exc}") from exc

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Run inference on multiple texts in a single batched forward pass.

        Args:
            texts: List of raw input strings.

        Returns:
            List of dicts with 'label' and 'confidence' keys, aligned with input order.

        Raises:
            PredictionError: If tokenisation or model inference fails unexpectedly.
        """
        with tracer.start_as_current_span("model.batch_inference") as span:
            span.set_attribute("batch.size", len(texts))
            try:
                # padding=True pads each sequence to the longest in this batch
                # rather than always padding to MAX_LENGTH, which saves compute
                # when most texts are short.
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                )

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # probs shape: [N, num_labels]
                probs = F.softmax(outputs.logits, dim=-1)
                predicted_indices = probs.argmax(dim=-1).tolist()
                # max probability per row == confidence of the predicted class
                confidences = probs.max(dim=-1).values.tolist()

                results = [
                    {
                        "label": LABELS[idx],
                        "confidence": round(conf, 4),
                    }
                    for idx, conf in zip(predicted_indices, confidences)
                ]
                span.set_attribute("batch.labels", str([r["label"] for r in results]))
                return results
            except (ModelNotLoadedError, PredictionError):
                raise
            except Exception as exc:
                raise PredictionError(f"Batch inference failed: {exc}") from exc


def load_all_versions(versions_dir: Path) -> dict[str, Predictor]:
    """Scan versions_dir for subdirectories containing model checkpoints and load each.

    Checkpoint dirs that lack tokenizer files share the tokenizer from the first
    version that has one (tokenizer is identical across training epochs).
    Directories that fail to load are skipped with a warning.
    """
    predictors: dict[str, Predictor] = {}
    if not versions_dir.exists():
        logger.warning("Model versions directory not found", path=str(versions_dir))
        return predictors

    candidates = sorted(p for p in versions_dir.iterdir() if p.is_dir() and (p / "config.json").exists())

    # Locate a shared tokenizer (first dir that has tokenizer_config.json).
    shared_tokenizer_dir = next(
        (p for p in candidates if (p / "tokenizer_config.json").exists()), None
    )

    for path in candidates:
        has_tokenizer = (path / "tokenizer_config.json").exists()
        tok_dir = path if has_tokenizer else shared_tokenizer_dir
        try:
            logger.info("Loading model version", version=path.name, path=str(path))
            predictors[path.name] = Predictor(artifact_dir=path, tokenizer_dir=tok_dir)
        except Exception as exc:
            logger.warning("Skipping model version", version=path.name, error=str(exc))

    return predictors
