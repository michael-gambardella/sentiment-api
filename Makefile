.PHONY: install train evaluate test serve clean \
        docker-build docker-up docker-down docker-logs

install:
	pip install -r requirements.txt

train:
	python -m model.trainer

evaluate:
	python -m model.evaluator

test:
	pytest tests/ -v

serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t sentiment-api:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api
