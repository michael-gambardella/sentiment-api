.PHONY: install train evaluate test serve clean

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
