"""SHAP token-level attribution for the sentiment classifier.

Uses SHAP's PartitionExplainer with a word-level Text masker. The masker splits
on non-word characters so attribution tokens stay human-readable regardless of
the model's internal subword tokenisation.

build_explainer is called once per model version at startup.
compute_attributions is synchronous and CPU-bound; route handlers must call it
via run_in_executor so the event loop stays unblocked during the many model
calls SHAP requires.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shap

if TYPE_CHECKING:
    from api.predictor import Predictor

_LABEL_TO_IDX = {"NEGATIVE": 0, "POSITIVE": 1}


def build_explainer(predictor: "Predictor") -> shap.Explainer:
    """Return a SHAP PartitionExplainer wrapping predictor.predict_batch.

    Construction is cheap (no inference runs). All computation happens
    when the returned explainer is called.
    """

    def _predict_proba(texts: list[str]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if not texts:
            return np.zeros((0, 2), dtype=np.float64)
        results = predictor.predict_batch(texts)
        out: list[list[float]] = []
        for r in results:
            p_pos = r["confidence"] if r["label"] == "POSITIVE" else 1.0 - r["confidence"]
            out.append([1.0 - p_pos, p_pos])
        return np.array(out, dtype=np.float64)

    masker = shap.maskers.Text(tokenizer=r"\W+")
    return shap.Explainer(_predict_proba, masker, output_names=["NEGATIVE", "POSITIVE"])


def compute_attributions(
    explainer: shap.Explainer,
    text: str,
    predicted_label: str,
) -> tuple[list[dict], float]:
    """Run SHAP and return (attributions, base_value) for predicted_label.

    attributions — list of {"token": str, "score": float} in input order.
                   Positive score pushes toward predicted_label; negative away.
    base_value   — P(predicted_label) when all tokens are masked (the baseline).

    The additive property holds: sum(scores) + base_value ≈ confidence.

    This function is synchronous. Call via asyncio.get_running_loop().run_in_executor
    from async route handlers.
    """
    label_idx = _LABEL_TO_IDX[predicted_label]
    sv = explainer([text])

    tokens: list[str] = sv.data[0]
    scores: np.ndarray = sv.values[0, :, label_idx]
    base_value = float(sv.base_values[0, label_idx])

    return (
        [{"token": str(tok), "score": round(float(s), 4)} for tok, s in zip(tokens, scores)],
        round(base_value, 4),
    )
