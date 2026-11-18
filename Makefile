PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install lint format test download ingest eval eval-generation api ui demo

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e . --no-deps

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:
	$(BIN)/pytest

download:
	$(BIN)/python scripts/download_corpus.py

ingest:
	$(BIN)/python scripts/ingest.py

eval:
	$(BIN)/python scripts/evaluate.py

eval-generation:
	$(BIN)/python scripts/evaluate.py --generation

api:
	$(BIN)/python -m swissdocs.api

ui:
	$(BIN)/streamlit run apps/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

demo:
	docker compose up --build
