.PHONY: setup data models export load-sql verify rebuild run app test lint check

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
	$(VENV_PY) -m src.expert

export:
	$(VENV_PY) -m src.export_sql

# Loads data/export/*.csv into SQL Server (owner-authorised exception; see src/load_sql.py).
# For --dry-run, --verify-only or --only <table> run python -m src.load_sql directly.
load-sql:
	$(VENV_PY) -m src.load_sql

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

# Every check, like `.	asks.ps1 verify`: runs all of them, then fails if any failed.
verify:
	@fail=0; 	$(VENV_PY) -m ruff check . || fail=1; 	$(VENV_PY) -m mypy src || fail=1; 	$(VENV_PY) -m pytest -q || fail=1; 	$(VENV_PY) -m src.load_sql --verify-only || fail=1; 	$(VENV_PY) scripts/make_results.py --check || fail=1; 	$(VENV_PY) scripts/app_smoke.py || fail=1; 	exit $$fail

# Full pipeline with the CSV export, reload SQL Server, then verify everything.
rebuild:
	$(VENV_PY) run_all.py --export
	$(VENV_PY) -m src.load_sql
	$(MAKE) verify
