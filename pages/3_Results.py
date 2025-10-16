"""Narrative results combining regression outputs with visual summaries."""
from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

from data_loader import load_model_table, parse_numeric

sns.set_theme(style="darkgrid")


PALETTE = [
    "#355070",
    "#6d597a",
    "#b56576",
    "#e56b6f",
    "#eaac8b",
    "#f9844a",
    "#f9c74f",
]


def _build_palette(length: int) -> list[str]:
    if length <= len(PALETTE):
        return PALETTE[:length]
    return sns.color_palette("husl", length)

st.title("Results")
st.caption("Regression results on fertility, values, and religiosity")

st.markdown(
    """
    The models below show fertility differentials across religious groups and value systems.
    Each subsection presents the original regression table together with focused visualisations
    that highlight the most policy-relevant patterns described in the analytical notes.
    """
)


TABLE_TITLE_OVERRIDES = {
    "Poisson_Muslim.xlsx": "Table 1. Poisson regression for number of children",
    "Poisson_Values.xlsx": "Table 2. Poisson regression with value indexes",
    "Poisson_relig.xlsx": "Table 3. Poisson regression for religiosity",
    "Cox_muslim.xlsx": "Table 4. Cox models for birth spacing",
    "Cox_rel.xlsx": "Table 5. Cox models for religiosity interactions",
}


def _resolve_title(file_name: str, raw_title: str, default: str) -> str:
    if file_name in TABLE_TITLE_OVERRIDES:
        return TABLE_TITLE_OVERRIDES[file_name]
    return raw_title or default


def _get_row(table: pd.DataFrame, variable_key: str) -> pd.Series:
    mask = table["Variable"].str.strip().str.lower() == variable_key.lower()
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def _format_spec_label(column: str, name_map: Dict[str, str] | None = None) -> Tuple[str, int]:
    match = re.match(r"\((\d+)\)\s*(.*)", column)
    if match:
        order = int(match.group(1))
        base = match.group(2).strip()
        base_key = base
        if name_map and base_key in name_map:
            friendly = name_map[base_key]
        else:
            friendly = base.replace("_", " ").title()
        return f"{order} – {friendly}", order
    return column, 0


def _build_effect_frame(
    table: pd.DataFrame,
    variable_map: Dict[str, str],
    *,
    spec_name_map: Dict[str, str] | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    records = []
    column_iterable: Iterable[str]
    if columns is None:
        column_iterable = [col for col in table.columns if col not in {"Variable", "Context"}]
    else:
        column_iterable = [col for col in columns if col in table.columns]

    for var_key, label in variable_map.items():
        row = _get_row(table, var_key)
        if row.empty:
            continue
        for column in column_iterable:
            value = parse_numeric(row.get(column, ""))
            if value is None:
                continue
            spec_label, order = _format_spec_label(column, spec_name_map)
            records.append(
                {
                    "Specification": spec_label,
                    "Order": order,
                    "Effect": value,
                    "Coefficient": label,
                }
            )
    return pd.DataFrame(records)


def _ordinal(number: int) -> str:
    suffix = "th"
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _build_birth_effect_frame(table: pd.DataFrame, variable_map: Dict[str, str], birth_columns: Iterable[str]) -> pd.DataFrame:
    records = []
    for var_key, label in variable_map.items():
        row = _get_row(table, var_key)
        if row.empty:
            continue
        for column in birth_columns:
            if column not in table.columns:
                continue
            value = parse_numeric(row.get(column, ""))
            if value is None:
                continue
            match = re.match(r"(\d+)_birth", column)
            if not match:
                continue
            number = int(match.group(1))
            records.append(
                {
                    "Birth order": f"{_ordinal(number)} birth",
                    "Order": number,
                    "Effect": value,
                    "Coefficient": label,
                }
            )
    return pd.DataFrame(records)


def _plot_effect_lines(effect_df: pd.DataFrame, y_label: str, title: str, *, baseline: float | None = 1.0) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values(["Order", "Coefficient"])
    categories = ordered.sort_values("Order")["Specification"].unique()
    ordered["Specification"] = pd.Categorical(ordered["Specification"], categories=categories, ordered=True)

    unique_coeffs = ordered["Coefficient"].unique()
    palette = _build_palette(len(unique_coeffs))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(unique_coeffs) == 1:
        sns.lineplot(
            data=ordered,
            x="Specification",
            y="Effect",
            color=palette[0],
            marker="o",
            linewidth=2.2,
            ax=ax,
            legend=False,
        )
    else:
        sns.lineplot(
            data=ordered,
            x="Specification",
            y="Effect",
            hue="Coefficient",
            palette=palette,
            marker="o",
            linewidth=2.2,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Coefficient")
            legend.set_frame_on(False)
            legend.set_bbox_to_anchor((1.02, 1))
    if baseline is not None:
        ax.axhline(baseline, color="#222222", linestyle="--", linewidth=1)
    for line in ax.lines:
        line.set_markeredgecolor("white")
        line.set_markeredgewidth(0.8)
        line.set_markersize(7)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Specification")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    y_min = ordered["Effect"].min()
    y_max = ordered["Effect"].max()
    if baseline is not None:
        y_min = min(y_min, baseline)
        y_max = max(y_max, baseline)
    span = y_max - y_min
    margin = 0.1 if span == 0 else span * 0.12
    ax.set_ylim(y_min - margin, y_max + margin)
    fig.tight_layout()
    return fig


def _plot_birth_effects(effect_df: pd.DataFrame, title: str, *, y_label: str = "Hazard ratio", baseline: float | None = 1.0) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values(["Order", "Coefficient"])
    categories = ordered.sort_values("Order")["Birth order"].unique()
    ordered["Birth order"] = pd.Categorical(ordered["Birth order"], categories=categories, ordered=True)

    unique_coeffs = ordered["Coefficient"].unique()
    palette = _build_palette(len(unique_coeffs))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(unique_coeffs) == 1:
        sns.lineplot(
            data=ordered,
            x="Birth order",
            y="Effect",
            color=palette[0],
            marker="o",
            linewidth=2.2,
            ax=ax,
            legend=False,
        )
    else:
        sns.lineplot(
            data=ordered,
            x="Birth order",
            y="Effect",
            hue="Coefficient",
            palette=palette,
            marker="o",
            linewidth=2.2,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Coefficient")
            legend.set_frame_on(False)
            legend.set_bbox_to_anchor((1.02, 1))
    if baseline is not None:
        ax.axhline(baseline, color="#222222", linestyle="--", linewidth=1)
    for line in ax.lines:
        line.set_markeredgecolor("white")
        line.set_markeredgewidth(0.8)
        line.set_markersize(7)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Birth order")
    ax.set_title(title)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    y_min = ordered["Effect"].min()
    y_max = ordered["Effect"].max()
    if baseline is not None:
        y_min = min(y_min, baseline)
        y_max = max(y_max, baseline)
    span = y_max - y_min
    margin = 0.1 if span == 0 else span * 0.12
    ax.set_ylim(y_min - margin, y_max + margin)
    fig.tight_layout()
    return fig


def _plot_category_bars(effect_df: pd.DataFrame, title: str, y_label: str, *, baseline: float | None = 1.0) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values("Level")
    colors = _build_palette(len(ordered))
    palette = dict(zip(ordered["Label"], colors))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=ordered, x="Label", y="Effect", palette=palette, ax=ax, saturation=0.9)
    if baseline is not None:
        ax.axhline(baseline, color="#222222", linestyle="--", linewidth=1)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Religiosity level")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    for container in ax.containers:
        ax.bar_label(container, fmt="{:.2f}")
    fig.tight_layout()
    return fig


def _plot_category_gradient(effect_df: pd.DataFrame, title: str, y_label: str, *, baseline: float | None = 1.0) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values("Level")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    base_color = PALETTE[0]
    ax.plot(ordered["Level"], ordered["Effect"], color=base_color, linewidth=2.2, marker="o")
    scatter_palette = dict(zip(ordered["Level"], _build_palette(len(ordered))))
    sns.scatterplot(
        data=ordered,
        x="Level",
        y="Effect",
        hue="Level",
        palette=scatter_palette,
        s=140,
        ax=ax,
        legend=False,
        edgecolor="white",
        linewidth=0.8,
    )
    if baseline is not None:
        ax.axhline(baseline, color="#222222", linestyle="--", linewidth=1)
    ax.set_xticks(ordered["Level"])
    ax.set_xticklabels(ordered["Label"], rotation=20, ha="right")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Religiosity intensity")
    ax.set_title(title)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    y_min = ordered["Effect"].min()
    y_max = ordered["Effect"].max()
    if baseline is not None:
        y_min = min(y_min, baseline)
        y_max = max(y_max, baseline)
    span = y_max - y_min
    margin = 0.1 if span == 0 else span * 0.12
    ax.set_ylim(y_min - margin, y_max + margin)
    fig.tight_layout()
    return fig


# --- Poisson models: denomination effects -----------------------------------
poisson_file = "Poisson_Muslim.xlsx"
poisson_title, poisson_table = load_model_table(poisson_file)
section_title = _resolve_title(poisson_file, poisson_title, "Poisson regression for number of children")
st.subheader(section_title)
st.dataframe(poisson_table, use_container_width=True)

poisson_spec_map = {
    "IRR_age": "Age controls",
    "IRR_reg": "Age + region",
    "IRR_full control - reg": "Full + region",
    "IRR_full control": "Full controls",
    "IRR_full control+occupation": "Full + occupation",
    "Other_Mus_IRR_age": "Age controls (extended)",
}
poisson_effects = _build_effect_frame(
    poisson_table,
    {"muslim": "Self-identified Muslim", "muslim1": "Extended Muslim (muslim1)"},
    spec_name_map=poisson_spec_map,
)
poisson_fig = _plot_effect_lines(
    poisson_effects,
    "Incidence rate ratio",
    "Muslim fertility advantage across specifications",
)
if poisson_fig:
    st.pyplot(poisson_fig)
    plt.close(poisson_fig)

st.markdown(
    """
    The Poisson regressions confirm a robust Muslim fertility premium even as richer
    sets of controls are introduced. The extended *muslim1* coding—which folds in the
    small Hindu cluster noted in the fieldwork—delivers slightly larger incidence rate
    ratios, echoing the narrative evidence from the modelling notes.
    """
)


# --- Poisson models: value indexes -------------------------------------------
values_file = "Poisson_Values.xlsx"
values_title, values_table = load_model_table(values_file)
st.subheader(_resolve_title(values_file, values_title, "Poisson regressions with value indexes"))
st.dataframe(values_table, use_container_width=True)

value_spec_map = {
    "Parental help": "Parental support (continuous)",
    "Children help": "Children support (continuous)",
    "Egalitarian": "Egalitarian values (continuous)",
    "Both": "All continuous indexes",
    "Egalitarian_short": "Egalitarian (binary)",
    "Family support_short": "Family support (binary)",
    "Both shorts": "All binary indexes",
}
base_value_effects = _build_effect_frame(
    values_table,
    {
        "parents_support": "Parental support",
        "children_support": "Children support",
        "egalitarian": "Egalitarian values",
        "family_support": "Family support composite",
    },
    spec_name_map=value_spec_map,
)
base_value_fig = _plot_effect_lines(
    base_value_effects,
    "Incidence rate ratio",
    "Family and egalitarian values in fertility models",
)
if base_value_fig:
    st.pyplot(base_value_fig)
    plt.close(base_value_fig)

interaction_value_effects = _build_effect_frame(
    values_table,
    {
        "1.muslim1#c.parents_support": "Parental support × Muslim",
        "1.muslim1#c.children_support": "Children support × Muslim",
        "1.muslim1#c.egalitarian": "Egalitarian × Muslim",
        "1.muslim1#c.family_support": "Family support × Muslim",
    },
    spec_name_map=value_spec_map,
)
interaction_value_fig = _plot_effect_lines(
    interaction_value_effects,
    "Incidence rate ratio",
    "Value interactions with Muslim identification",
)
if interaction_value_fig:
    st.pyplot(interaction_value_fig)
    plt.close(interaction_value_fig)

st.markdown(
    """
    Consistent with the descriptive notes, stronger family-support norms raise the
    expected number of children, whereas egalitarian attitudes dampen fertility. The
    interaction plot highlights how family support pays a larger dividend within the
    Muslim population, while egalitarianism suppresses fertility for both faith groups.
    """
)


# --- Poisson models: religiosity ---------------------------------------------
relig_file = "Poisson_relig.xlsx"
relig_title, relig_table = load_model_table(relig_file)
st.subheader(_resolve_title(relig_file, relig_title, "Poisson regressions for religiosity"))
st.dataframe(relig_table, use_container_width=True)

category_records = []
religiosity_labels = {
    2: "Low religiosity",
    3: "Moderate religiosity",
    4: "High religiosity",
    5: "Very high religiosity",
}
for level, label in religiosity_labels.items():
    row = _get_row(relig_table, f"recode of a1112 (religiousity) = {level}")
    if row.empty:
        continue
    value = parse_numeric(row.get("(1) Religiuos cat", ""))
    if value is None:
        continue
    category_records.append({"Level": level, "Label": f"{label} (cat. {level})", "Effect": value})

category_df = pd.DataFrame(category_records)
category_fig = _plot_category_bars(
    category_df,
    "Fertility differences by religiosity intensity",
    "Incidence rate ratio",
)
if category_fig:
    st.pyplot(category_fig)
    plt.close(category_fig)

category_gradient_fig = _plot_category_gradient(
    category_df,
    "Gradient of fertility effects across religiosity levels",
    "Incidence rate ratio",
)
if category_gradient_fig:
    st.pyplot(category_gradient_fig)
    plt.close(category_gradient_fig)

continuous_row = _get_row(relig_table, "religiosity")
continuous_effect = parse_numeric(continuous_row.get("(2) Religious", "")) if not continuous_row.empty else None
if continuous_effect is not None:
    st.metric("Effect of continuous religiosity index", f"IRR = {continuous_effect:.3f}")

st.markdown(
    """
    Rising religiosity—whether measured categorically or through the continuous index—
    is associated with higher completed fertility. Even moderately religious respondents
    exceed the baseline, aligning with the qualitative inference that any degree of
    religiosity supports larger family ideals.
    """
)


# --- Cox models: birth timing by denomination --------------------------------
cox_muslim_file = "Cox_muslim.xlsx"
cox_muslim_title, cox_muslim_table = load_model_table(cox_muslim_file)
st.subheader(_resolve_title(cox_muslim_file, cox_muslim_title, "Cox models for birth spacing"))
st.dataframe(cox_muslim_table, use_container_width=True)

birth_columns = [col for col in cox_muslim_table.columns if col.endswith("birth .")]
cox_muslim_effects = _build_birth_effect_frame(cox_muslim_table, {"muslim1": "Muslim households"}, birth_columns)
cox_muslim_fig = _plot_birth_effects(
    cox_muslim_effects,
    "Progression to higher birth orders for Muslim households",
)
if cox_muslim_fig:
    st.pyplot(cox_muslim_fig)
    plt.close(cox_muslim_fig)

st.markdown(
    """
    The timing models echo the count regressions: Muslim households transition more
    quickly to every subsequent birth, with the most pronounced gap at the third child.
    This pattern dovetails with the model commentary that motivated the deeper dive into
    value-based mechanisms.
    """
)


# --- Cox models: religiosity interactions ------------------------------------
cox_rel_file = "Cox_rel.xlsx"
cox_rel_title, cox_rel_table = load_model_table(cox_rel_file)
st.subheader(_resolve_title(cox_rel_file, cox_rel_title, "Cox models for religiosity"))
st.dataframe(cox_rel_table, use_container_width=True)

rel_birth_columns = [col for col in cox_rel_table.columns if col.endswith("birth .")]
base_rel_effects = _build_birth_effect_frame(
    cox_rel_table,
    {
        "2.religious": "Religiosity level 2",
        "3.religious": "Religiosity level 3",
        "4.religious": "Religiosity level 4",
        "5.religious": "Religiosity level 5",
    },
    rel_birth_columns,
)
base_rel_fig = _plot_birth_effects(
    base_rel_effects,
    "Hazard ratios by religiosity intensity",
)
if base_rel_fig:
    st.pyplot(base_rel_fig)
    plt.close(base_rel_fig)

interaction_rel_effects = _build_birth_effect_frame(
    cox_rel_table,
    {
        "1.muslim1": "Muslim baseline",
        "1.muslim1#2.religious": "Muslim × level 2",
        "1.muslim1#3.religious": "Muslim × level 3",
        "1.muslim1#4.religious": "Muslim × level 4",
        "1.muslim1#5.religious": "Muslim × level 5",
    },
    rel_birth_columns,
)
interaction_rel_fig = _plot_birth_effects(
    interaction_rel_effects,
    "Muslim advantage conditional on religiosity",
)
if interaction_rel_fig:
    st.pyplot(interaction_rel_fig)
    plt.close(interaction_rel_fig)

st.markdown(
    """
    Among non-Muslim respondents, stronger religiosity raises the likelihood of first
    births but loses traction for later parities. The interaction plot shows that Muslim
    households sustain elevated hazards even at comparable religiosity levels, matching
    the narrative that religiosity amplifies fertility primarily within the Muslim group.
    """
)
