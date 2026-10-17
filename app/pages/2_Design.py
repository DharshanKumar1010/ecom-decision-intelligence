"""Page 2 — Design: the late-delivery models, what drives them, and sensitivity (Unit 3).

Honesty note: every quality number here comes from order-grouped evaluation (no order is in
both train and test): the held-out metrics in `reports/model_metrics.json` and the
out-of-fold metrics in `reports/oof_metrics.json`. `LatePredictions.parquet` is out-of-fold, but
this page still does not derive lift, gain or "catches X% of late orders" from it; the
temporal check (`reports/diagnostics/diagnostic_temporal.txt`) shows the models have little skill on
future periods, so no claim here goes beyond ranking quality on unseen orders.
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
    report_path,
    temporal_results,
)
from app._components import (
    caption,
    footer,
    kpis,
    note,
    page_header,
    run_page,
    section,
    set_page,
    show,
    subtitle,
    table,
    technical,
)
from app._format import fmt_num, fmt_pct
from src import config, explain

set_page("Design")
_SUBTITLE = page_header("How likely is an order to arrive late, and why?")

# key in model_metrics.json -> (display name, name used for the grouped-CV summary)
_MODELS = (
    ("random_forest", "Random forest", "RandomForest"),
    ("decision_tree", "Decision tree", "DecisionTree"),
    ("logistic_regression", "Linear model", "LogisticRegression"),
)
_SCALES = ("-25%", "-10%", "+10%", "+25%")
_FLIP = "flip_0_1"


def _cv() -> dict[str, tuple[float, float]]:
    """Out-of-fold ROC-AUC mean and std across the order-grouped folds, per model."""
    if not report_path("oof_metrics.json").exists():
        return {}
    models = cast(dict[str, Any], load_report_json("oof_metrics.json"))["models"]
    names = {key: cv_name for key, _, cv_name in _MODELS}
    return {
        names[key]: (float(m["oof_fold_auc_mean"]), float(m["oof_fold_auc_std"]))
        for key, m in models.items() if key in names
    }


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


def _tradeoff_sentence(metrics: dict[str, Any]) -> str:
    aucs = {key: float(metrics[key]["roc_auc"]) for key, _, _ in _MODELS}
    names = {key: display.lower() for key, display, _ in _MODELS}
    best = max(aucs, key=lambda key: aucs[key])
    others = [k for k in aucs if k != "random_forest"]
    size = config.MODELS_DIR / "random_forest.joblib"
    storage = (
        f" The saved forest takes about {size.stat().st_size / 1e6:,.0f} MB on disk."
        if size.exists()
        else ""
    )
    scores = ", ".join(f"{aucs[k] * 100:.0f}% for the {names[k]}" for k in others)
    if best == "random_forest":
        lead = (
            "The random forest is the primary model because it ranks late orders best on "
            f"orders it never saw ({aucs['random_forest'] * 100:.0f}%, against {scores}). "
        )
    else:
        lead = (
            "The random forest is the primary model, but on orders it never saw the "
            f"{names[best]} now ranks at least as well ({aucs[best] * 100:.0f}% against "
            f"{aucs['random_forest'] * 100:.0f}% for the forest). "
        )
    return lead + (
        "The decision tree and the linear model can be read step by step, so they are kept "
        "for explanation." + storage
    )


_TOP_EFFECTS = 5
# A forward-looking AUC below this is read as "no better than chance".
_NEAR_CHANCE = 0.55
# Multiplicative nudges to calendar numbers (month 1-12, weekday 0-6) are not realistic
# scenarios, so they are kept out of the short chart; the full heatmap still shows them.
_CALENDAR_INPUTS = ("order_month", "day_of_week")


def _largest_effects(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """For the primary model, each input's single largest nudge effect; the top few overall."""
    rf = sensitivity[
        (sensitivity["model"] == "random_forest") & ~sensitivity["feature"].isin(_CALENDAR_INPUTS)
    ].copy()
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
            marker={"color": [theme.RISK_RAISES if v > 0 else theme.RISK_LOWERS
                              for v in top["mean_late_risk_change"]]},
            text=[fmt_num(v, 3, signed=True) for v in top["mean_late_risk_change"]],
            textposition="outside", cliponaxis=False,
            customdata=[fmt_num(v, 4, signed=True) for v in top["mean_late_risk_change"]],
            hovertemplate="%{y}: %{customdata}<extra></extra>",
        )
    )
    limit = float(top["mean_late_risk_change"].abs().max()) * 1.4
    fig.update_layout(
        height=260,
        xaxis={"title": "Average change in late risk (red = higher risk)",
               "range": [-limit, limit], "showgrid": True, "gridcolor": theme.GRID,
               "zeroline": True, "zerolinecolor": theme.TEXT_MUTED},
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
    common = {"zmid": 0, "zmin": -limit, "zmax": limit, "colorscale": theme.RISK_SCALE}
    pretty = grid.map(lambda v: fmt_num(v, 3, signed=True)).to_numpy()
    exact = grid.map(lambda v: fmt_num(v, 4, signed=True)).to_numpy()
    flip_text = [[fmt_num(v, 3, signed=True) for v in flips["mean_late_risk_change"]]]
    flip_exact = [[fmt_num(v, 4, signed=True) for v in flips["mean_late_risk_change"]]]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.86, 0.14], vertical_spacing=0.12)
    fig.add_trace(
        go.Heatmap(
            z=grid.to_numpy(), x=list(grid.columns), y=grid.index.tolist(),
            text=pretty, texttemplate="%{text}", customdata=exact,
            hovertemplate="%{y}<br>%{x}: %{customdata}<extra></extra>",
            colorbar={"title": "Change in late risk<br>(red = higher risk)", "len": 0.9,
                      "y": 0.55}, **common,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=[flips["mean_late_risk_change"].tolist()], x=["flip 0 → 1"],
            y=[explain.feature_label(f) for f in flips["feature"]][:1] or [""],
            text=flip_text, texttemplate="%{text}", customdata=flip_exact, showscale=False,
            hovertemplate="%{y}: %{customdata}<extra></extra>", **common,
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
        (raises, "Raises late risk", theme.RISK_RAISES),
        (~raises, "Lowers late risk", theme.RISK_LOWERS),
    ):
        part = coefficients[mask]
        fig.add_trace(go.Bar(
            x=part["standardized_coefficient"], y=part["label"], orientation="h",
            name=name, marker={"color": color},
            customdata=[fmt_num(v, 3, signed=True) for v in part["standardized_coefficient"]],
            hovertemplate="%{y}: %{customdata}<extra></extra>",
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
        "chart shows how far the predicted late risk moves on average (red = higher risk, "
        "blue = lower risk)."
    )
    show(_heatmap(sensitivity, model))
    note("The yes/no input (seller and customer in the same state) is flipped from 0 to 1 in "
         "the bottom row. Month of purchase and day of the week are scaled as plain numbers "
         "(1 to 12 and 0 to 6) like every other input, which is a stress test, not a "
         "realistic scenario.")


def _split_note(rf: dict[str, Any]) -> str:
    """How the test fold was made and the threshold behind the late-class scores."""
    split = cast(dict[str, Any], load_report_json("oof_metrics.json"))["split"]
    return (
        "Order-grouped split: no order appears in both training and test (this is fold 0 of "
        f"{config.PREDICT_N_FOLDS}; seed {config.SEED}). The test fold holds "
        f"{int(split['test_items']):,} items from {int(split['test_orders']):,} orders; the late "
        f"rate in it is {fmt_pct(float(rf['base_rate']), 2)}. Precision, recall and F1 are for "
        "the late class at the library's default decision threshold: an item is predicted late "
        "when the model's score is 0.5 or more. ROC-AUC does not depend on a threshold and "
        "leads because the target is imbalanced."
    )


def _confusion_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """The three confusion matrices as ONE table: a row per model and actual outcome."""
    rows = []
    for key, display, _ in _MODELS:
        matrix = metrics[key]["confusion_matrix"]
        for actual, counts in zip(("On time", "Late"), matrix, strict=True):
            rows.append({
                "Model": display, "Actual": actual,
                "Predicted on time": counts[0], "Predicted late": counts[1],
            })
    return pd.DataFrame(rows)


def _technical(metrics: dict[str, Any], cv: dict[str, tuple[float, float]],
               importance: pd.DataFrame) -> None:
    rf = metrics["random_forest"]
    st.markdown("**Model comparison (held-out test fold)**")
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
            row["Order-grouped 5-fold ROC-AUC"] = (
                f"{fmt_num(cv[cv_name][0], 4)} ± {fmt_num(cv[cv_name][1], 4)}"
            )
        rows.append(row)
    table(pd.DataFrame(rows), {"ROC-AUC": "num4", "Precision (late)": "num4",
                               "Recall (late)": "num4", "F1 (late)": "num4"})
    note(_split_note(rf))
    if "RandomForest" in cv:
        st.markdown(explain.cv_sentence(*cv["RandomForest"], folds=config.PREDICT_N_FOLDS))

    st.markdown("**Confusion matrices (counts of test items)**")
    table(_confusion_table(metrics), {"Predicted on time": "int", "Predicted late": "int"})

    st.markdown("**Permutation importance**")
    shown = importance.assign(feature=importance["feature"].map(explain.feature_label))
    table(shown, {"importance_mean": "num4", "importance_std": "num4"})
    st.markdown("**Decision tree rule path (depth 4)**")
    st.markdown("Class 1 = late, class 0 = on time.")
    st.code(load_report_text("decision_tree.txt"), language="text")
    st.markdown("**Linear model weights**")
    _coefficients()
    st.markdown("**Sensitivity analysis**")
    _sensitivity()


def _strongest_input_sentence(importance: pd.DataFrame) -> str:
    """The strongest input by permutation importance, with a caveat when it is a calendar input."""
    top = str(importance.sort_values("importance_mean").iloc[-1]["feature"])
    label = explain.feature_label(top)
    sentence = f"{label} is the strongest input."
    if top in _CALENDAR_INPUTS:
        sentence += (
            " That may reflect particular busy or disrupted periods in this dataset rather "
            "than anything that repeats."
        )
    return sentence


def _forward_looking_sentence() -> str:
    """What the forward-looking test showed, from the temporal diagnostic report (or nothing)."""
    results = temporal_results()
    if not results:
        return ""
    best = max(results["all"].values())
    close_to_chance = best < _NEAR_CHANCE
    verdict = (
        "no model did meaningfully better than chance" if close_to_chance
        else "the best model did better than chance"
    )
    tail = (
        " So the score ranks sellers and orders within the period we have; it is not a forecast."
        if close_to_chance else ""
    )
    return (
        "On a forward-looking test (trained on earlier orders, tested on later ones) "
        f"{verdict}: the best ROC-AUC was {fmt_num(best, 2)}, where 0.50 is a coin flip."
        + tail
    )


def render() -> None:
    metrics = cast(dict[str, Any], load_report_json("model_metrics.json"))
    rf = metrics["random_forest"]
    auc, base_rate = float(rf["roc_auc"]), float(rf["base_rate"])
    importance = load_report_csv("rf_feature_importance.csv")
    cv = _cv()

    subtitle(_SUBTITLE, explain.design_answer(auc, base_rate))
    kpis([
        ("Late-delivery rate", fmt_pct(base_rate, 1),
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
    st.markdown(_tradeoff_sentence(metrics))
    st.markdown(f"{_strongest_input_sentence(importance)} {_forward_looking_sentence()}".strip())

    section("What if an input changes?")
    show(_effects_chart(_largest_effects(load_report_csv("sensitivity.csv"))))
    caption(f"Random forest: average change in predicted late risk when one input is nudged "
            f"and the others are held fixed. The {_TOP_EFFECTS} largest effects are shown; month "
            "and weekday are left out because scaling a calendar number is not a realistic "
            "scenario. Full view in Technical details.")

    with technical():
        _technical(metrics, cv, importance)


run_page(render)
footer(
    "Predictive modeling and sensitivity analysis (Unit 3)",
    "Predictive",
    "Tactical",
)
