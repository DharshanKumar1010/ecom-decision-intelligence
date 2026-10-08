"""Page 6 — DSS Architecture: phases of decision making and DSS components (Unit 2)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app._common import DATA_CATALOG, load_knowledge_base
from app._components import (
    footer,
    kpis,
    note,
    page_header,
    run_page,
    section,
    set_page,
    subtitle,
    technical,
)
from src import config, explain

set_page("DSS Architecture")
_SUBTITLE = page_header("How does the system fit together?")

# CLAUDE.md section 2: DSS subsystem mapping.
_SUBSYSTEMS = (
    ("Data management", "Builds, stores and serves the data",
     ("src/build_tables.py", "src/synthetic.py", "src/export_sql.py")),
    ("Model management", "Analytical and predictive models",
     ("src/clickstream.py", "src/sentiment.py", "src/predict.py", "src/ahp.py",
      "src/discovery.py")),
    ("Knowledge base", "Expert rules and the inference/explanation engine",
     ("src/expert.py", "rules/seller_rules.yaml")),
    ("User interface", "How decision makers see and use the results",
     ("app/Home.py", "app/_common.py", "app/_components.py", "app/_theme.py",
      "app/pages/1_Intelligence.py", "app/pages/2_Design.py", "app/pages/3_Choice.py",
      "app/pages/4_Discovery.py", "app/pages/5_Implementation.py",
      "app/pages/6_DSS_Architecture.py")),
)

# Simon's phases of decision making -> page(s).
_PHASES = (
    ("Intelligence", "Find the problem", "Intelligence"),
    ("Design", "Model it", "Design"),
    ("Choice", "Evaluate and select", "Choice, with Discovery as a companion"),
    ("Implementation", "Act and explain", "Implementation"),
)


def _flow(n_tables: int, n_rules: int) -> None:
    boxes = (
        ("1. Data", [f"{n_tables} data tables", "Real Olist orders, reviews and sellers",
                     "Synthetic clickstream and calls", "Cleaned and checked by the pipeline"]),
        ("2. Models", ["Late-delivery risk (random forest)", "Seller ranking (AHP)",
                       "Quality vs popularity groups", "Call sentiment (VADER)"]),
        ("3. Rules", [f"{n_rules} readable if-then rules",
                      "Keep, Warn, Suspend, Feature, Promote",
                      "The highest-priority rule wins", "Every answer comes with a reason"]),
        ("4. People", ["This Streamlit app", "Power BI report (built by hand)",
                       "Managers see scores and reasons", "Analysts can open the details"]),
    )
    for col, (title, lines) in zip(st.columns(4), boxes, strict=True):
        body = "".join(f"<p>{line}</p>" for line in lines)
        col.markdown(f'<div class="flow-box"><h4>{title}</h4>{body}</div>',
                     unsafe_allow_html=True)
    st.markdown(
        '<div class="flow-note">Read left to right: data feeds models, models feed rules, and '
        "rules reach people. SQL Server and Power BI are a downstream layer built by hand from "
        "the exported CSVs.</div>",
        unsafe_allow_html=True,
    )


def _catalog() -> None:
    rows = []
    for entry in DATA_CATALOG:
        count = entry.row_count()
        rows.append({
            "Table": entry.name,
            "Location": ("data/processed/" if entry.kind == "parquet" else "reports/")
                        + entry.path.name,
            "Rows": f"{count:,}" if count is not None else "—",
            "Status": "ok" if count is not None else "missing — run `python run_all.py`",
            "Synthetic": "Synthetic data" if entry.synthetic else "",
            "Description": entry.description,
        })
    frame = pd.DataFrame(rows)
    n_missing = int((frame["Status"] != "ok").sum())
    if n_missing:
        st.markdown(
            f"**{n_missing} of {len(frame)} catalog files are missing.** Run "
            "`python run_all.py` from the project root to generate them."
        )
    st.markdown(f"**Data catalog: all {len(frame)} tables (row counts read live)**")
    st.table(frame.set_index("Table").style.set_properties(
        subset=["Rows"], **{"text-align": "right"}
    ))


def _modules() -> None:
    rows = []
    for name, role, files in _SUBSYSTEMS:
        missing = [f for f in files if not (config.PROJECT_ROOT / f).exists()]
        rows.append({
            "Subsystem": name,
            "Role": role,
            "Modules": ", ".join(files),
            "All present": "Yes" if not missing else f"MISSING: {', '.join(missing)}",
        })
    st.markdown("**Subsystems and the modules that implement them**")
    st.table(pd.DataFrame(rows).set_index("Subsystem"))


def render() -> None:
    n_rules = len(load_knowledge_base().rules)
    subtitle(_SUBTITLE, explain.architecture_answer(len(DATA_CATALOG), n_rules))
    kpis([
        ("Data tables", f"{len(DATA_CATALOG)}", None),
        ("Expert rules", f"{n_rules}", None),
    ])
    _flow(len(DATA_CATALOG), n_rules)
    section("Phases of decision making")
    st.table(pd.DataFrame(_PHASES, columns=["Phase", "Question it answers", "Page"])
             .set_index("Phase"))

    with technical():
        _catalog()
        _modules()
        note("Rows marked Synthetic data are generated (Olist has no real clickstream or "
             "call-centre data).")


run_page(render)
footer(
    "Phases of decision making and DSS components (Unit 2)",
    "Descriptive",
    "Strategic",
)
