# Sentiment-Aware Text Classification API

A production-style REST API that classifies the sentiment of text as **POSITIVE** or **NEGATIVE** using a fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased) transformer model.

Built as a portfolio project demonstrating end-to-end ML engineering: data pipeline, model training, API deployment, and testing.

---

## What It Does

Send a piece of text to the `/predict` endpoint and get back a sentiment label with a confidence score:

```json
POST /predict
{ "text": "This movie was absolutely fantastic." }

→ { "label": "POSITIVE", "confidence": 0.97 }
```

---

## Architecture

```
Raw Text (IMDB Dataset)
        ↓
data/pipeline.py       ← ETL: clean → tokenize → split → DataLoader
        ↓
model/trainer.py       ← Fine-tune DistilBERT (AdamW + LR scheduler)
        ↓
model/artifacts/       ← Saved model weights + tokenizer
        ↓
api/predictor.py       ← Load model once at startup, run inference
        ↓
api/main.py            ← FastAPI: /predict, /health, /metrics
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | `distilbert-base-uncased` via Hugging Face Transformers |
| Training | PyTorch + AdamW optimizer |
| Data | Hugging Face `datasets` (IMDB, 50k samples) |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Testing | pytest + httpx |

---

## Project Structure

```
sentiment-api/
├── data/
│   └── pipeline.py        # ETL: load, clean, tokenize, split
├── model/
│   ├── trainer.py         # Fine-tuning loop with checkpointing
│   ├── evaluator.py       # Accuracy, F1, confusion matrix
│   └── artifacts/         # Saved weights (gitignored)
├── api/
│   ├── main.py            # FastAPI app with lifespan model loading
│   ├── schemas.py         # Pydantic request/response models
│   └── predictor.py       # Inference logic with softmax confidence
├── tests/
│   ├── test_pipeline.py   # Data shape and cleaning assertions
│   ├── test_model.py      # Model output shape and label correctness
│   └── test_api.py        # Full API contract via TestClient
├── notebooks/
│   └── exploration.ipynb  # EDA + training curves
├── requirements.txt
└── Makefile
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# 2. Install dependencies
make install

# 3. Train the model (~25 min on GPU, longer on CPU)
make train

# 4. Run the test suite
make test

# 5. Start the API server
make serve
```

The server starts at `http://localhost:8000`. Interactive API docs are available at `http://localhost:8000/docs`.

---

## API Usage

### POST /predict

Classify the sentiment of a text string.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This was one of the best films I have ever seen."}'
```

```json
{"label": "POSITIVE", "confidence": 0.9731}
```

### GET /health

Check that the service is running and the model is loaded.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

### GET /metrics

Return metadata about the loaded model.

```bash
curl http://localhost:8000/metrics
```

```json
{
  "model_name": "distilbert-base-uncased",
  "artifact_path": "/path/to/model/artifacts/final",
  "max_input_length": 256,
  "labels": ["NEGATIVE", "POSITIVE"]
}
```

---

## Training Results

Fine-tuned `distilbert-base-uncased` for 3 epochs on the IMDB dataset (25,000 train / 25,000 test).

| Metric | Score |
|---|---|
| Accuracy | **91.60%** |
| F1 Score | **91.65%** |

**Confusion matrix (25,000 test samples):**

|  | Predicted NEGATIVE | Predicted POSITIVE |
|---|---|---|
| **Actual NEGATIVE** | 11,385 ✓ | 1,115 ✗ |
| **Actual POSITIVE** | 984 ✗ | 11,516 ✓ |

The model is slightly better at identifying positive reviews (recall 92.1%) than negative ones (recall 91.1%). The confusion matrix is saved to `model/artifacts/confusion_matrix.png`.

---

## Architecture Decisions

**Why DistilBERT instead of full BERT?**
DistilBERT is 40% smaller and 60% faster at inference while retaining 97% of BERT's performance on NLP benchmarks. For a binary sentiment classifier, the trade-off is straightforward — there is no meaningful accuracy benefit from using the larger model on this task.

**Why MAX_LENGTH = 256 instead of 512?**
DistilBERT supports up to 512 tokens, but IMDB reviews average around 230 tokens. Setting the cap at 256 covers the vast majority of reviews while halving memory usage compared to the maximum. Longer reviews are truncated, but the most sentiment-relevant content typically appears early in a review.

**Why is the model loaded at startup, not per request?**
Loading a 250MB checkpoint takes approximately 2 seconds. Loading it inside the prediction function would add that latency to every API call. FastAPI's `lifespan` context manager loads the model once before the first request is accepted and holds it in `app.state` for the lifetime of the process.

**Why AdamW at a learning rate of 2e-5?**
AdamW decouples weight decay from the adaptive gradient update, which is important for transformer fine-tuning. Standard Adam applies weight decay incorrectly through the gradient, which distorts regularization. The 2e-5 learning rate follows the recommendation from the original BERT paper — low enough to update the pre-trained weights without overwriting the language knowledge they encode (catastrophic forgetting).

**Why a linear warmup + decay schedule?**
The warmup phase gradually increases the learning rate from 0 to peak over the first 10% of training steps. This prevents large gradient updates before the model has stabilized from the random initialization of the classification head. Linear decay then reduces the learning rate to 0 by the final step, allowing finer adjustments as the model converges.

**Why validate inputs with Pydantic before reaching the model?**
An empty string or a missing field would pass through the tokenizer and produce a technically valid but meaningless prediction. Enforcing `min_length=1` and `max_length=5000` at the schema level rejects bad inputs immediately with a 422 response, before any compute is spent on tokenization or inference.
