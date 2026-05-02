.PHONY: install lint format test data train serve docker-build docker-run compose-up compose-down k8s-apply helm-install eda clean

install:
	pip install -r requirements.txt

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

test:
	pytest --cov=src --cov-report=term-missing

data:
	python -m src.data.download

train:
	python -m src.models.train

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t heart-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-api:latest

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down -v

k8s-apply:
	kubectl apply -f deploy/k8s/

helm-install:
	helm upgrade --install heart-api deploy/helm/heart-api

eda:
	jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov mlruns models/*.joblib
