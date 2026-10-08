"""Page 2 — Design: the late-delivery models, what drives them, and sensitivity (Unit 3).

Honesty note: `LatePredictions.parquet` is scored by models fitted on ~80% of those very
rows, so it is largely in-sample. This page therefore never derives lift, gain or "catches
X% of late orders" from it; every quality number here comes from the held-out metrics in
`reports/model_metrics.json` and the cross-validation report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app import _theme as theme
from app._common import (
    load_report_csv,
    load_report_json,
    load_report_text,
    parse_cv_summary,
    report_path,
)
from app._components import (
    caption,
    footer,
    kpis,
    note,
    page_header,
    review_line,
    run_page,
    section,
    set_page,
    show,
    subtitle,
    technical,
)
from src import config, explain

set_page("Design")
_SUBTITLE = page_header("How likely is an order to arrive late, and why?")

# key in model_metrics.json -> (display name, name in diagnostic_rf_cv.txt)
_MODELS = (
    ("random_forest", "Random forest", "RandomForest"),
    ("decision_tree", "Decision tree", "DecisionTree"),
    ("logistic_regression", "Linear model", "LogisticRegression"),
)
_SCALES = ("-25%", "-10%", "+10%", "+25%")
_FLIP = "flip_0_1"


def _cv() -> dict[str, tuple[float, float]]:
    if report_path("diagnostic_rf_cv.txt").exists():
        return parse_cv_summary(load_report_text("diagnostic_rf_cv.txt"))
    return {}


def _importance_chart(importance: pd.DataFrame) -> go.Figure:
    ordered = importance.sort_values("importance_mean")
    fig = go.Figure(
        go.Bar(
            x=ordered["importance_mean"],
            y=[explain.feature_label(f) for f in ordered["feature"]],
            orientation="h",
            marker={"color": theme.ACCENT},
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        xaxis={"title": "How much worse the model gets when this input is scrambled",
               "showgrid": True, "gridcolor": theme.GRID},
        yaxis={"showgrid": False},
    )
    return fig


def _tradeoff_sentence() -> str:
    size = config.MODELS_DIR / "random_forest.joblib"
    storage = (
        f" The saved forest takes about {size.stat().st_size / 1e6:,.0f} MB on disk."
        if size.exists()
        else ""
    )
    return (
        "The random forest is the primary model because it ranks late orders best. The "
        "decision tree and the linear model rank less well but can be read step by step, "
        "so they are kept for explanation." + storage
    )


_TOP_EFFECTS = 5


def _largest_effects(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """For the primary model, each input's single largest nudge effect; the top few overall."""
    rf = sensitivity[sensitivity["model"] == "random_forest"].copy()
    rf["abs_change"] = rf["mean_late_risk_change"].abs()
    biggest = rf.sort_values("abs_change", ascending=False).drop_duplicates("feature")
    top = biggest.head(_TOP_EFFECTS).sort_values("mean_late_risk_change")
    nudge = top["perturbation"].map(lambda p: "flip 0 → 1" if p == _FLIP else p)
    top["label"] = [
        f"{explain.feature_label(f)} ({n})" for f, n in zip(top["feature"], nudge, strict=True)
    ]
    return top


def _effects_chart(top: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=top["mean_late_risk_change"], y=top["label"], orientation="h",
            marker={"color": theme.ACCENT},
            text=[f"{v:+.3f}" for v in top["mean_late_risk_change"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    limit = float(top["mean_late_risk_change"].abs().max()) * 1.4
    fig.update_layout(
        height=260,
        xaxis={"title": "Average change in predicted late risk", "range": [-limit, limit],
               "showgrid": True, "gridcolor": theme.GRID, "zeroline": True,
               "zerolinecolor": theme.TEXT_MUTED},
        yaxis={"showgrid": False},
        margin={"l": 8, "r": 40, "t": 8, "b": 8},
    )
    return fig


def _heatmap(sensitivity: pd.DataFrame, model: str) -> go.Figure:
    subset = sensitivity[sensitivity["model"] == model]
    scaled = subset[subset["perturbation"].isin(_SCALES)]
    grid = scaled.pivot(index="feature", columns="perturbation", values="mean_late_risk_change")
    grid = grid[[c for c in _SCALES if c in grid.columns]]
    grid.index = [explain.feature_label(f) for f in grid.index]
    flips = subset[subset["perturbation"] == _FLIP]
    limit = float(max(grid.abs().max().max(), flips["mean_late_risk_change"].abs().max(), 1e-9))
    common = {"zmid": 0, "zmin": -limit, "zmax": limit, "colorscale": theme.DIVERGING_SCALE}

    fig = make_subplots(rows=2, cols=1, row_heights=[0.86, 0.14], vertical_spacing=0.12)
    fig.add_trace(
        go.Heatmap(
            z=grid.to_numpy(), x=list(grid.columns), y=grid.index.tolist(),
            texttemplate="%{z:+.3f}", hovertemplate="%{y}<br>%{x}: %{z:+.4f}<extra></extra>",
            colorbar={"title": "Change in<br>late risk", "len": 0.9, "y": 0.55}, **common,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=[flips["mean_late_risk_change"].tolist()], x=["flip 0 → 1"],
            y=[explain.feature_label(f) for f in flips["feature"]][:1] or [""],
            texttemplate="%{z:+.3f}", showscale=False,
            hovertemplate="%{y}: %{z:+.4f}<extra></extra>", **common,
        ),
        row=2, col=1,
    )
    fig.update_xaxes(type="category", side="top", showgrid=False, row=1, col=1)
    fig.update_xaxes(type="category", showgrid=False, row=2, col=1)
    fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    fig.update_layout(height=520, margin={"l": 8, "r": 8, "t": 28, "b": 8})
    return fig


def _coefficients() -> None:
    coefficients = load_report_csv("coefficients.csv").copy()
    coefficients["label"] = coefficients["feature"].map(explain.feature_label)
    coefficients = coefficients.sort_values("standardized_coefficient")
    raises = coefficients["standardized_coefficient"] > 0
    fig = go.Figure()
    for mask, name, color in (
        (raises, "Raises late risk", theme.ACTIONS["Suspend"].color),
        (~raises, "Lowers late risk", theme.QUADRANTS["Star"].color),
    ):
        part = coefficients[mask]
        fig.add_trace(go.Bar(
            x=part["standardized_coefficient"], y=part["label"], orientation="h",
            name=name, marker={"color": color},
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        ))
    fig.update_layout(
        height=420, barmode="relative",
        xaxis={"title": "Weight (a bigger bar pushes the prediction further)",
               "showgrid": True, "gridcolor": theme.GRID},
        yaxis={"showgrid": False},
    )
    show(fig)
    note("Month of purchase and day of the week enter this model as plain numbers (1 to 12, "
         "0 to 6), so its weights for them do not describe seasonality.")


def _sensitivity() -> None:
    sensitivity = load_report_csv("sensitivity.csv")
    names = {key: display for key, display, _ in _MODELS}
    available = [k for k, _, _ in _MODELS if k in set(sensitivity["model"])]
    model = st.selectbox(
        "Model", available, format_func=lambda k: names[k], key="design_sensitivity_model"
    )
    st.markdown(
        "Each input is nudged by -25%, -10%, +10% and +25% while the others stay fixed; the "
        "chart shows how far the predicted late risk moves on average."
    )
    show(_heatmap(sensitivity, model))
    note("The yes/no input (seller and customer in the same state) is flipped from 0 to 1 in "
         "the bottom row. Scaling calendar inputs such as the month is a stress test, not a "
         "realistic scenario.")


def _technical(metrics: dict[str, Any], cv: dict[str, tuple[float, float]],
               importance: pd.DataFrame) -> None:
    rf = metrics["random_forest"]
    st.markdown("**Model comparison (held-out 20% test set)**")
    rows = []
    for key, display, cv_name in _MODELS:
        m = metrics[key]
        row: dict[str, Any] = {
            "Model": display,
            "ROC-AUC": m["roc_auc"],
            "Precision (late)": m["precision"]["1"],
            "Recall (late)": m["recall"]["1"],
            "F1 (late)": m["f1"]["1"],
        }
        if cv_name in cv:
            row["5-fold CV ROC-AUC"] = f"{cv[cv_name][0]:.4f} ± {cv[cv_name][1]:.4f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    note(f"Stratified split, seed {config.SEED}; target is_late with base rate "
         f"{float(rf['base_rate']) * 100:.2f}%. ROC-AUC leads because the target is imbalanced.")
    if "RandomForest" in cv:
        st.markdown(explain.cv_sentence(*cv["RandomForest"], folds=5))

    st.markdown("**Confusion matrices (rows actual, columns predicted)**")
    for col, (key, display, _) in zip(st.columns(3), _MODELS, strict=True):
        col.markdown(display)
        col.dataframe(
            pd.DataFrame(
                metrics[key]["confusion_matrix"],
                index=["actual on time", "actual late"],
                columns=["predicted on time", "predicted late"],
            ),
            width="stretch",
        )

    st.markdown("**Permutation importance (raw column names)**")
    st.dataframe(importance, hide_index=True, width="stretch")
    st.markdown("**Decision tree rule path (depth 4)**")
    st.code(load_report_text("decision_tree.txt"), language="text")
    st.markdown("**Linear model weights**")
    _coefficients()
    st.markdown("**Sensitivity analysis**")
    _sensitivity()


def render() -> None:
    metrics = cast(dict[str, Any], load_report_json("model_metrics.json"))
    rf = metrics["random_forest"]
    auc, base_rate = float(rf["roc_auc"]), float(rf["base_rate"])
    importance = load_report_csv("rf_feature_importance.csv")
    cv = _cv()

    subtitle(_SUBTITLE, explain.design_answer(auc, base_rate))
    review_line()
    kpis([
        ("Late-delivery rate", f"{base_rate * 100:.1f}%",
         "Share of order items that arrive after the estimated date."),
        ("Ranking score", f"{auc * 100:.0f}%", explain.auc_sentence(auc)),
        ("Inputs used", f"{len(config.ALLOWED_FEATURES)}",
         "Facts about an order that the models use to estimate its late risk."),
    ])
    section("What drives late deliveries")
    show(_importance_chart(importance))
    caption(explain.top_drivers_sentence(
        dict(zip(importance["feature"], importance["importance_mean"], strict=True))
    ))
    st.markdown(_tradeoff_sentence())

    section("What if an input changes?")
    show(_effects_chart(_largest_effects(load_report_csv("sensitivity.csv"))))
    caption("Random forest, held-out test set: the average change in predicted late risk (a "
            "probability from 0 to 1) when one input is nudged and the others are held fixed. "
            f"The {_TOP_EFFECTS} largest effects are shown; the full view is in Technical "
            "details.")

    with technical():
        _technical(metrics, cv, importance)


run_page(render)
footer(
    "Predictive modeling and sensitivity analysis (Unit 3)",
    "Predictive",
    "Tactical",
)
