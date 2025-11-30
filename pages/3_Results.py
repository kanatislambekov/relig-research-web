"""Narrative results combining regression outputs with visual summaries."""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

from data_loader import load_model_table, parse_numeric

FONT_FAMILY = "Helvetica"
FONT_RC = {
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
}

plt.rcParams.update({"font.family": FONT_FAMILY, **FONT_RC})
sns.set_theme(style="whitegrid", font=FONT_FAMILY, rc=FONT_RC)


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


def _ratio_limits(values: Iterable[float], baseline: float | None, *, pad: float = 0.12) -> tuple[float, float]:
    """Return axis bounds that start at 1.0 when ratios stay above one."""
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return (0.9, 1.1) if baseline == 1.0 else (0, 1)
    vmin = float(series.min())
    vmax = float(series.max())
    if baseline is not None:
        vmin = min(vmin, baseline)
        vmax = max(vmax, baseline)
    span = vmax - vmin
    margin = 0.1 if span == 0 else span * pad
    lower = vmin - margin
    upper = vmax + margin
    if baseline == 1.0 and vmin >= 1.0:
        lower = 1.0
    return lower, upper


_FIGURE_NUMBER = {"value": 1}


def _figure_caption(title: str, description: str) -> None:
    """Render a numbered caption beneath a figure."""
    number = _FIGURE_NUMBER["value"]
    _FIGURE_NUMBER["value"] += 1
    st.markdown(f"**Figure {number}. {title}.** {description.strip()}")


def _add_baseline_tick(ax: plt.Axes, *, axis: str, baseline: float | None) -> None:
    """Ensure the axis shows the baseline (e.g., IRR=1.0) as an explicit tick."""
    if baseline is None:
        return
    if axis == "y":
        ticks = list(ax.get_yticks())
        ticks.append(baseline)
        ticks = sorted(set(ticks))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2f}" if abs(tick) < 10 else f"{tick:g}" for tick in ticks])
    elif axis == "x":
        ticks = list(ax.get_xticks())
        ticks.append(baseline)
        ticks = sorted(set(ticks))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.2f}" if abs(tick) < 10 else f"{tick:g}" for tick in ticks])

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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    ax.set_ylim(lower, upper)
    _add_baseline_tick(ax, axis="y", baseline=baseline)
    fig.tight_layout()
    return fig


def _plot_effect_horizontal(
    effect_df: pd.DataFrame,
    x_label: str,
    title: str,
    *,
    baseline: float | None = 1.0,
    min_limit: float | None = None,
    max_limit: float | None = None,
) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values(["Order", "Coefficient"])
    categories = ordered.sort_values("Order")["Specification"].unique()
    ordered["Specification"] = pd.Categorical(ordered["Specification"], categories=categories, ordered=True)
    positions = np.arange(len(categories))
    spec_to_pos = {spec: pos for pos, spec in enumerate(categories)}
    ordered["Spec position"] = ordered["Specification"].map(spec_to_pos)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    unique_coeffs = ordered["Coefficient"].unique()
    palette = _build_palette(len(unique_coeffs))
    for color, coeff in zip(palette, unique_coeffs):
        subset = ordered[ordered["Coefficient"] == coeff]
        if subset.empty:
            continue
        subset = subset.sort_values("Spec position")
        ax.plot(
            subset["Effect"],
            subset["Spec position"],
            marker="o",
            linewidth=2,
            color=color,
            label=coeff,
        )
    if baseline is not None:
        ax.axvline(baseline, color="#222222", linestyle="--", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Specification")
    ax.legend(loc="upper left", frameon=False)

    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    if min_limit is not None:
        lower = min_limit
    if max_limit is not None:
        upper = max_limit
    ax.set_xlim(lower, upper)
    _add_baseline_tick(ax, axis="x", baseline=baseline)

    for line in ax.lines:
        line.set_markeredgecolor("white")
        line.set_markeredgewidth(0.8)
        line.set_markersize(7)

    fig.tight_layout()
    return fig


def _plot_effect_small_multiples(effect_df: pd.DataFrame, axis_label: str, title: str, *, baseline: float | None = 1.0) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values(["Order", "Coefficient"]).copy()
    ordered["Specification"] = ordered["Specification"].astype(str)

    coefficients = ordered["Coefficient"].unique()
    total = len(coefficients)
    cols = 2 if total > 1 else 1
    rows = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6 + 0.8, rows * 3.4 + 0.4), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).reshape(-1)
    palette = _build_palette(total)

    for idx, (coeff, color) in enumerate(zip(coefficients, palette)):
        ax = axes_array[idx]
        subset = ordered[ordered["Coefficient"] == coeff].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset = subset.sort_values("Order").reset_index(drop=True)
        positions = np.arange(len(subset))
        ax.plot(
            subset["Effect"],
            positions,
            marker="o",
            linewidth=2,
            color=color,
        )
        if baseline is not None:
            ax.axvline(baseline, color="#3c3c3c", linestyle="--", linewidth=0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels(subset["Specification"])
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        ax.set_xlabel("")
        for line in ax.lines:
            line.set_markeredgecolor("white")
            line.set_markeredgewidth(0.8)
            line.set_markersize(6)

    for ax in axes_array[total:]:
        ax.axis("off")

    fig.supxlabel(axis_label)
    fig.supylabel("Specification")

    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    for ax in axes_array[:total]:
        ax.set_xlim(lower, upper)
        _add_baseline_tick(ax, axis="x", baseline=baseline)

    fig.tight_layout(rect=(0.03, 0.02, 1, 0.93))
    return fig


def _plot_birth_effects(
    effect_df: pd.DataFrame,
    title: str,
    *,
    y_label: str = "Hazard ratio",
    baseline: float | None = 1.0,
    force_zero_min: bool = False,
    upper_cap: float | None = None,
) -> plt.Figure | None:
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
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    if force_zero_min:
        lower = 0
    if upper_cap is not None:
        upper = upper_cap
    ax.set_ylim(lower, upper)
    _add_baseline_tick(ax, axis="y", baseline=baseline)
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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    for container in ax.containers:
        ax.bar_label(container, fmt="{:.2f}")
    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    ax.set_ylim(lower, upper)
    _add_baseline_tick(ax, axis="y", baseline=baseline)
    fig.tight_layout()
    return fig


def _plot_category_lollipops(
    effect_df: pd.DataFrame,
    title: str,
    x_label: str,
    *,
    baseline: float | None = 1.0,
    force_zero_min: bool = False,
    upper_cap: float | None = None,
) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values("Level")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    positions = np.arange(len(ordered))
    ax.hlines(positions, baseline if baseline is not None else ordered["Effect"].min(), ordered["Effect"], color="#94a3b8", linewidth=2)
    ax.scatter(ordered["Effect"], positions, color="#1d4ed8", s=80, zorder=3, edgecolor="white", linewidth=0.8)
    if baseline is not None:
        ax.axvline(baseline, color="#222222", linestyle="--", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["Label"])
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    if force_zero_min:
        lower = 0
    if upper_cap is not None:
        upper = upper_cap
    ax.set_xlim(lower, upper)
    _add_baseline_tick(ax, axis="x", baseline=baseline)
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
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    ax.set_ylim(lower, upper)
    _add_baseline_tick(ax, axis="y", baseline=baseline)
    fig.tight_layout()
    return fig


def _plot_value_spec_lines(effect_df: pd.DataFrame, title: str, y_label: str, *, baseline: float | None = 1.0) -> plt.Figure | None:
    """Line chart of value coefficients across specifications."""
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values(["Order", "Coefficient"])
    categories = ordered.sort_values("Order")["Specification"].unique()
    ordered["Specification"] = pd.Categorical(ordered["Specification"], categories=categories, ordered=True)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    palette = _build_palette(ordered["Coefficient"].nunique())
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
    if baseline is not None:
        ax.axhline(baseline, color="#222222", linestyle="--", linewidth=1)
    for line in ax.lines:
        line.set_markeredgecolor("white")
        line.set_markeredgewidth(0.8)
        line.set_markersize(7)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Specification")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    ax.set_ylim(lower, upper)
    _add_baseline_tick(ax, axis="y", baseline=baseline)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Coefficient")
        legend.set_frame_on(False)
        legend.set_bbox_to_anchor((1.02, 1))
    fig.tight_layout()
    return fig


def _plot_value_interaction_bars(effect_df: pd.DataFrame, title: str, y_label: str, *, baseline: float | None = 1.0) -> plt.Figure | None:
    """Grouped bar chart for interaction effects across specifications."""
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values(["Order", "Coefficient"])
    categories = ordered.sort_values("Order")["Specification"].unique()
    ordered["Specification"] = pd.Categorical(ordered["Specification"], categories=categories, ordered=True)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    palette = _build_palette(ordered["Coefficient"].nunique())
    sns.barplot(
        data=ordered,
        x="Specification",
        y="Effect",
        hue="Coefficient",
        palette=palette,
        ax=ax,
        saturation=0.85,
    )
    if baseline is not None:
        ax.axhline(baseline, color="#222222", linestyle="--", linewidth=1)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Specification")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    lower, upper = _ratio_limits(ordered["Effect"], baseline)
    ax.set_ylim(lower, upper)
    _add_baseline_tick(ax, axis="y", baseline=baseline)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Interaction term")
        legend.set_frame_on(False)
        legend.set_bbox_to_anchor((1.02, 1))
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
poisson_fig = _plot_effect_horizontal(
    poisson_effects,
    "Incidence rate ratio",
    "Muslim fertility advantage across specifications",
    min_limit=0,
    max_limit=1.65,
)
if poisson_fig:
    ax_poisson = poisson_fig.axes[0] if poisson_fig.axes else None
    if ax_poisson is not None:
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        ax_poisson.set_xticks(ticks)
        ax_poisson.set_xticklabels([f"{tick:.1f}" for tick in ticks])
        ax_poisson.set_xlim(0, 1.65)
        ax_poisson.tick_params(axis="x", labelsize=12)
    st.pyplot(poisson_fig)
    plt.close(poisson_fig)
    _figure_caption(
        "Muslim fertility advantage across Poisson specifications",
        """
        Points plot incidence rate ratios (IRRs) for the Muslim coefficients across model variants;
        the vertical dashed line at 1.0 marks parity with the non-Muslim baseline. All IRRs sit above 1,
        showing a robust fertility premium even after adding controls for region and occupation.
        """,
    )

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
    "Parental help": "Parental support index",
    "Children help": "Children support index",
    "Egalitarian": "Egalitarian values index",
    "Both": "Full model: all continuous indexes",
    "Egalitarian_short": "Egalitarian indicator",
    "Family support_short": "Family support indicator",
    "Both shorts": "Full model: both binary indexes",
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
if not base_value_effects.empty:
    latest_order = base_value_effects["Order"].max()
    latest_spec = base_value_effects[base_value_effects["Order"] == latest_order].copy()
    latest_spec["Level"] = np.arange(1, len(latest_spec) + 1)
    latest_spec = latest_spec.rename(columns={"Coefficient": "Label"})
    latest_spec = latest_spec[["Level", "Label", "Effect"]]

interaction_value_fig = _plot_category_lollipops(
    interaction_value_effects.assign(Level=np.arange(1, len(interaction_value_effects) + 1), Label=interaction_value_effects["Coefficient"])[["Level", "Label", "Effect"]],
    "Interactions of Muslim identification with value indexes",
    "Incidence rate ratio",
    baseline=1.0,
    force_zero_min=True,
    upper_cap=1.2,
)
if interaction_value_fig:
    st.pyplot(interaction_value_fig)
    plt.close(interaction_value_fig)
    _figure_caption(
        "Interaction effects: values × Muslim identification",
        """
        Each lollipop plots the incidence rate ratio for a value–Muslim interaction term in the richest model.
        Lines start at zero and extend to the estimated IRR, highlighting which interactions boost fertility
        above the neutral threshold of 1.0.
        """,
    )

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
category_fig = _plot_category_lollipops(
    category_df,
    "Fertility differences by religiosity intensity",
    "Incidence rate ratio",
    baseline=1.0,
    force_zero_min=True,
    upper_cap=1.2,
)
if category_fig:
    st.pyplot(category_fig)
    plt.close(category_fig)
    _figure_caption(
        "Incidence rate ratios by religiosity category",
        """
        Lollipop markers compare IRRs for each religiosity category against the neutral threshold of 1.0.
        The horizontal line at 1.0 denotes no fertility difference relative to the baseline; markers to the
        right indicate higher completed fertility, underscoring the positive gradient associated with greater
        religiosity.
        """,
    )

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
cox_muslim_upper_cap = None
if not cox_muslim_effects.empty:
    cap_candidate = cox_muslim_effects["Effect"].max()
    if cap_candidate is not None and np.isfinite(cap_candidate):
        cox_muslim_upper_cap = max(cap_candidate, 1.0) + 0.2
cox_muslim_fig = _plot_birth_effects(
    cox_muslim_effects,
    "Progression to higher birth orders for Muslim households",
    force_zero_min=True,
    upper_cap=cox_muslim_upper_cap,
)
if cox_muslim_fig:
    st.pyplot(cox_muslim_fig)
    plt.close(cox_muslim_fig)
    _figure_caption(
        "Hazard ratios for progressing to subsequent births (Muslim vs. others)",
        """
        Lines plot hazard ratios for Muslim households at each birth order; the dashed line at 1.0 marks
        identical pacing to the comparison group. Ratios above 1.0 show faster transitions, with the largest
        acceleration appearing around the third child.
        """,
    )

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
base_rel_upper_cap = None
if not base_rel_effects.empty:
    cap_candidate = base_rel_effects["Effect"].max()
    if cap_candidate is not None and np.isfinite(cap_candidate):
        base_rel_upper_cap = max(cap_candidate, 1.0) + 0.2
base_rel_fig = _plot_birth_effects(
    base_rel_effects,
    "Hazard ratios by religiosity intensity",
    force_zero_min=True,
    upper_cap=base_rel_upper_cap,
)
if base_rel_fig:
    st.pyplot(base_rel_fig)
    plt.close(base_rel_fig)
    _figure_caption(
        "Birth timing by religiosity intensity among non-Muslims",
        """
        Hazard ratios greater than 1.0 indicate faster progression to the next birth relative to the baseline.
        The plot shows that modest religiosity lifts the likelihood of moving to higher parities, while the
        effect tapers for later births and higher intensity levels.
        """,
    )

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
interaction_upper_cap = None
if not interaction_rel_effects.empty:
    cap_candidate = interaction_rel_effects["Effect"].max()
    if cap_candidate is not None and np.isfinite(cap_candidate):
        interaction_upper_cap = max(cap_candidate, 1.0) + 0.2
interaction_rel_fig = _plot_birth_effects(
    interaction_rel_effects,
    "Muslim advantage conditional on religiosity",
    force_zero_min=True,
    upper_cap=interaction_upper_cap,
)
if interaction_rel_fig:
    st.pyplot(interaction_rel_fig)
    plt.close(interaction_rel_fig)
    _figure_caption(
        "Combined effects of Muslim identity and religiosity on birth timing",
        """
        Hazard ratios compare Muslim households at each religiosity level with the non-Muslim baseline.
        Ratios above 1.0 signal quicker transitions to subsequent births, indicating that Muslim identity
        retains an advantage even after conditioning on intensity of belief.
        """,
    )

st.markdown(
    """
    Among non-Muslim respondents, stronger religiosity raises the likelihood of first
    births but loses traction for later parities. The interaction plot shows that Muslim
    households sustain elevated hazards even at comparable religiosity levels, matching
    the narrative that religiosity amplifies fertility primarily within the Muslim group.
    """
)
