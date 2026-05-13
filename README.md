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
api/predictor.py       ← Load model once, run inference
        ↓
api/main.py            ← FastAPI: /predict, /health, /metrics
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | `distilbert-base-uncased` via Hugging Face Transformers |
| Training | PyTorch + AdamW optimizer |
| Data | Hugging Face `datasets` (IMDB) |
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
│   ├── main.py            # FastAPI app
│   ├── schemas.py         # Pydantic request/response models
│   └── predictor.py       # Inference logic
├── tests/
│   ├── test_pipeline.py
│   ├── test_model.py
│   └── test_api.py
├── notebooks/
│   └── exploration.ipynb  # EDA + training curves
├── requirements.txt
└── Makefile
```

---

## Setup

```bash
# Install dependencies
make install

# Train the model (~25 min on GPU, longer on CPU)
make train

# Run the test suite
make test

# Start the API server
make serve
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Classify sentiment of input text |
| `GET` | `/health` | Service health check |
| `GET` | `/metrics` | Model metadata and performance stats |

---

## Training Results

> *To be updated after training is complete.*

---

## Status

> **In progress** — actively being built.
