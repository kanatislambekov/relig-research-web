"""Interactive heat map for exploring regression effect sizes."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import count_stars, load_model_table, parse_numeric


st.title("Heat map explorer")
st.caption("Compare effect sizes across regression models and specifications using an interactive 2D grid.")


TABLE_FILES: Dict[str, Dict[str, str]] = {
    "poisson_muslim": {"file": "Poisson_Muslim.xlsx", "label": "Poisson – fertility (Muslim focus)"},
    "poisson_values": {"file": "Poisson_Values.xlsx", "label": "Poisson – fertility (Values)"},
    "poisson_relig": {"file": "Poisson_relig.xlsx", "label": "Poisson – religiosity"},
    "cox_muslim": {"file": "Cox_muslim.xlsx", "label": "Cox – birth spacing (Muslim focus)"},
    "cox_relig": {"file": "Cox_rel.xlsx", "label": "Cox – birth spacing (Religiosity)"},
}


def _parse_metric_metadata(column: str) -> Tuple[int | None, str, str]:
    """Return (order, readable label, metric group) for a regression column."""

    label = str(column).strip()
    order: int | None = None
    match = re.match(r"\((\d+)\)\s*(.*)", label)
    if match:
        order = int(match.group(1))
        label = match.group(2).strip() or label

    upper = label.upper()
    if "IRR" in upper:
        metric_group = "Incidence rate ratio"
    elif "COEF" in upper or "COEFFICIENT" in upper:
        metric_group = "Coefficient"
    elif "HAZARD" in upper or re.search(r"\bHR\b", upper):
        metric_group = "Hazard ratio"
    elif "ODDS" in upper or re.search(r"\bOR\b", upper):
        metric_group = "Odds ratio"
    else:
        metric_group = "Model output"

    return order, label, metric_group


@lru_cache(maxsize=None)
def _collect_effects() -> pd.DataFrame:
    """Aggregate tidy regression outputs across all result tables."""

    records: list[dict[str, object]] = []
    for table_key, meta in TABLE_FILES.items():
        _, table = load_model_table(meta["file"])
        if table.empty:
            continue

        for _, row in table.iterrows():
            variable = str(row.get("Variable", "")).strip()
            if not variable or "std. err" in variable.lower():
                continue

            context = str(row.get("Context", "")).strip() or "All outcomes"

            for column, cell in row.items():
                if column in {"Variable", "Context"}:
                    continue

                cell_str = str(cell).strip()
                if not cell_str:
                    continue

                value = parse_numeric(cell_str)
                if value is None:
                    continue

                order, label, metric_group = _parse_metric_metadata(column)
                records.append(
                    {
                        "table_key": table_key,
                        "table_label": meta["label"],
                        "variable": variable,
                        "context": context,
                        "metric_raw": column,
                        "metric_label": label,
                        "metric_group": metric_group,
                        "metric_order": order if order is not None else np.nan,
                        "value": value,
                        "significance": count_stars(cell_str),
                        "cell_display": cell_str,
                    }
                )

    effects = pd.DataFrame.from_records(records)
    if effects.empty:
        return effects

    # Ensure consistent ordering for metrics with numeric prefixes.
    effects["metric_order"] = effects["metric_order"].fillna(1e6)
    effects = effects.sort_values(["table_key", "metric_group", "metric_order", "metric_label", "variable"])

    signed_log = np.sign(effects["value"]) * np.log(np.where(effects["value"] == 0, np.nan, np.abs(effects["value"])) )
    effects["log_value"] = pd.Series(signed_log, index=effects.index).fillna(0)
    effects["distance_from_one"] = (effects["value"] - 1.0).abs()
    return effects


effects = _collect_effects()

if effects.empty:
    st.warning("No regression coefficients were found. Please upload the regression tables to the Results folder.")
    st.stop()


st.sidebar.header("Controls")

table_options = {key: meta["label"] for key, meta in TABLE_FILES.items() if key in effects["table_key"].unique()}
selected_table = st.sidebar.selectbox("Regression table", options=list(table_options.keys()), format_func=lambda key: table_options[key])

table_data = effects[effects["table_key"] == selected_table].copy()

metric_groups = sorted(table_data["metric_group"].unique())
default_groups = metric_groups if len(metric_groups) < 4 else metric_groups[:3]
selected_groups = st.sidebar.multiselect("Regression metric", options=metric_groups, default=default_groups)

if selected_groups:
    table_data = table_data[table_data["metric_group"].isin(selected_groups)]

contexts = sorted(table_data["context"].unique())
selected_contexts = st.sidebar.multiselect("Outcome context", options=contexts, default=contexts)
if selected_contexts:
    table_data = table_data[table_data["context"].isin(selected_contexts)]

if table_data.empty:
    st.info("No coefficients match the selected filters. Try broadening the selection.")
    st.stop()

significance_threshold = st.sidebar.slider("Minimum significance (stars)", 0, 3, 0, help="Filter coefficients by the number of significance stars.")
if significance_threshold:
    table_data = table_data[table_data["significance"] >= significance_threshold]

if table_data.empty:
    st.info("No coefficients remain after applying the significance filter.")
    st.stop()

z_options = {
    "Effect value": ("value", "Effect size"),
    "Log effect": ("log_value", "Log(effect)", "Logarithm of the effect size"),
    "Distance from neutral": ("distance_from_one", "|Effect − 1|", "Distance from the neutral value of 1"),
}

z_choice = st.sidebar.selectbox("Z dimension", options=list(z_options.keys()))
z_column = z_options[z_choice][0]
z_axis_title = z_options[z_choice][1]

# --- 2D heatmap ---
heatmap_source = table_data.pivot_table(index="variable", columns="metric_label", values=z_column, aggfunc="mean")
heatmap_source = heatmap_source.sort_index()

if not heatmap_source.empty:
    z_matrix = heatmap_source.to_numpy(dtype=float)
    finite_values = z_matrix[np.isfinite(z_matrix)]
    if finite_values.size == 0:
        st.info("The selected filters only contain missing values for the requested metric.")
        st.stop()

    z_min = np.nanmin(finite_values)
    z_max = np.nanmax(finite_values)
    # Introduce slight padding for the color scale
    margin = (z_max - z_min) * 0.05 if z_max != z_min else 0.05
    z_min -= margin
    z_max += margin

    base_heatmap = go.Heatmap(
        z=z_matrix,
        x=heatmap_source.columns,
        y=heatmap_source.index,
        colorscale="Viridis",
        colorbar=dict(title=z_axis_title),
        hoverongaps=False,
        zmin=z_min,
        zmax=z_max,
        xgap=2,
        ygap=2,
    )

    nan_overlay = go.Heatmap(
        z=np.where(np.isnan(z_matrix), 1, 0),
        x=heatmap_source.columns,
        y=heatmap_source.index,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "#b3b3b3"]],
        showscale=False,
        hoverinfo="skip",
        zmin=0,
        zmax=1,
        xgap=2,
        ygap=2,
    )

    heatmap_fig = go.Figure(data=[base_heatmap, nan_overlay])
    heatmap_fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        title=dict(text=f"2D projection – {table_options[selected_table]}", x=0.5),
        xaxis_title="Model / specification",
        yaxis_title="Variable",
        xaxis=dict(tickangle=0, automargin=True),
        yaxis=dict(automargin=True),
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)
else:
    st.info("Not enough data to construct the 2D heat map for the current filters.")


st.markdown(
    """
    **How to read the visualisation**

    * Each cell represents one coefficient drawn from the selected regression table.
    * Cell colours follow the selected vertical axis encoding (value, log value, or distance from 1).
    * Grey cells indicate specifications without data for that metric.
    * Use the filters in the sidebar to compare specific outcome contexts or specification blocks.
    """
)
