.PHONY: up down mlflow-ui prefect-ui install

up:
	docker compose up -d mlflow prefect

down:
	docker compose down

mlflow-ui:
	@echo "MLflow -> http://localhost:5050"

prefect-ui:
	@echo "Prefect -> http://localhost:4200"

install:
	python -m pip install -U pip
	pip install -r requirements.txt
