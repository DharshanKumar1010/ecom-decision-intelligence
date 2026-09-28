# E-Commerce Decision Support System

BIA 21CSE421T course project. A Python + Streamlit decision support system on the Olist Brazilian
e-commerce dataset that answers: **which sellers should the marketplace keep, warn, suspend, or
feature, and why?**

See [CLAUDE.md](CLAUDE.md) for the full specification. This README is a stub and will be filled in
(problem statement, architecture, setup, results, limitations) once the full pipeline exists.

## Status

Stage 1 (scaffolding + data layer) only: `src/config.py`, `src/build_tables.py`,
`src/synthetic.py`, `src/clickstream.py`, and their tests. Models, AHP, the expert system, and the
Streamlit app are not yet implemented.

## Setup and commands

This project is developed on Windows. Both a `Makefile` (for WSL/git-bash/CI) and a `tasks.ps1`
(for native PowerShell) expose the same targets:

| Task | Makefile (WSL / git-bash / CI) | PowerShell |
|---|---|---|
| Create venv, install deps | `make setup` | `.\tasks.ps1 setup` |
| Build data tables | `make data` | `.\tasks.ps1 data` |
| Train models | `make models` | `.\tasks.ps1 models` |
| Run full pipeline | `make run` | `.\tasks.ps1 run` |
| Launch dashboard | `make app` | `.\tasks.ps1 app` |
| Run tests | `make test` | `.\tasks.ps1 test` |
| Lint + type-check | `make lint` | `.\tasks.ps1 lint` |
| Lint + test (pre-done-check) | `make check` | `.\tasks.ps1 check` |

`make models`, `make run`, and `make app` will fail until later stages (`sentiment.py`,
`predict.py`, `ahp.py`, `expert.py`, `run_all.py`, `app/`) are implemented — expected at this stage.

Place the 9 raw Olist CSVs in `data/raw/` before running `data`/`run` (see CLAUDE.md section 6.1).
