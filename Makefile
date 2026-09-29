.PHONY: setup data models run app test lint check

VENV_PY := .venv/bin/python
ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
endif

setup:
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

data:
	$(VENV_PY) -m src.build_tables
	$(VENV_PY) -m src.synthetic

models:
	$(VENV_PY) -m src.sentiment
	$(VENV_PY) -m src.predict
	$(VENV_PY) -m src.ahp
	$(VENV_PY) -m src.discovery

run:
	$(VENV_PY) run_all.py

app:
	$(VENV_PY) -m streamlit run app/Home.py

test:
	$(VENV_PY) -m pytest -q

lint:
	$(VENV_PY) -m ruff check .
	$(VENV_PY) -m mypy src

check: lint test
