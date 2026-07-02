install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	black src/ tests/ && isort src/ tests/ --profile black && flake8 src/ tests/ && mypy src/

test:
	pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

security:
	bandit -r src/ -ll -ii

complexity:
	radon cc src/ -nc

docstrings:
	interrogate src/ --fail-under=80

audit:
	pip-audit -r requirements.txt && detect-secrets scan --baseline .secrets.baseline

train-baseline:
	python -m src.baseline.train_baseline

train:
	python -m src.training.train

evaluate:
	python -m src.evaluation.evaluate

redteam:
	python -m src.evaluation.redteam

serve:
	uvicorn src.api.app:app --reload --port 8000

gradio:
	python -m src.ui.gradio_app

docker-build:
	docker build -t p1-hybrid-jailbreak-detector .
