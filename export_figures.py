"""Generate PNG snapshots of key figures (excluding heatmaps) with bold captions."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")  # Off-screen rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker as mticker

from data_loader import load_kaz_ggs, load_model_table, parse_numeric

# --- Global styling ---------------------------------------------------------
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

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


def _annotate_and_save(fig: plt.Figure, slug: str, caption: str) -> None:
    """Write a bold caption onto the figure and save to PNG."""
    fig.text(
        0.01,
        -0.06,
        caption,
        ha="left",
        va="top",
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        fontsize=11,
    )
    fig.savefig(EXPORT_DIR / f"{slug}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Data & Methods figures -------------------------------------------------
def export_sampling_chart() -> None:
    survey = load_kaz_ggs()
    region_column = "aregion" if "aregion" in survey.columns else None
    if not region_column:
        return
    region_counts = survey[region_column].astype(str).value_counts().head(10).sort_values(ascending=False)
    if region_counts.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=region_counts.values, y=region_counts.index, color="#3a7ca5", ax=ax)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("Respondents", fontweight="bold")
    ax.set_ylabel("Region", fontweight="bold")
    ax.set_xlim(left=0)
    _annotate_and_save(fig, "fig_sampling", "Figure: Survey coverage across regions")


def export_covariate_heatmap() -> None:
    survey = load_kaz_ggs()
    covariate_columns = [column for column in ("aage", "numbiol", "family_support", "egalitarian") if column in survey.columns]
    if not covariate_columns:
        return
    covariate_frame = survey[covariate_columns].apply(pd.to_numeric, errors="coerce")
    numeric_covariates = covariate_frame.dropna(axis=0, how="any")
    if numeric_covariates.empty:
        return
    corr = numeric_covariates.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("", fontweight="bold")
    ax.set_ylabel("", fontweight="bold")
    _annotate_and_save(fig, "fig_covariates", "Figure: Correlation structure of continuous covariates")


# --- Descriptive statistics figures ----------------------------------------
POST_SOVIET_COUNTRIES = [
    ("Armenia", "ARM"),
    ("Azerbaijan", "AZE"),
    ("Belarus", "BLR"),
    ("Estonia", "EST"),
    ("Georgia", "GEO"),
    ("Kazakhstan", "KAZ"),
    ("Kyrgyzstan", "KGZ"),
    ("Latvia", "LVA"),
    ("Lithuania", "LTU"),
    ("Moldova", "MDA"),
    ("Russia", "RUS"),
    ("Tajikistan", "TJK"),
    ("Turkmenistan", "TKM"),
    ("Ukraine", "UKR"),
    ("Uzbekistan", "UZB"),
]


def _load_post_soviet_tfr(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def export_tfr(path: Path) -> None:
    frame = _load_post_soviet_tfr(path)
    if frame.empty:
        print(f"Skipping TFR export: file not found at {path}", file=sys.stderr)
        return
    recent_years = frame["Year"].unique()
    if recent_years.size > 0:
        max_year = int(recent_years.max())
        min_year = max(max_year - 10, int(frame["Year"].min()))
        plot_frame = frame[frame["Year"].between(min_year, max_year)].copy()
    else:
        plot_frame = frame.copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    highlight_color = "#1b8a5a"
    base_color = "#404040"
    replacement_level = 2.1
    east_europe_group = {"Belarus", "Estonia", "Latvia", "Lithuania", "Moldova", "Russia", "Ukraine"}
    east_endpoints: list[tuple[float, float]] = []

    for country, code in POST_SOVIET_COUNTRIES:
        country_rows = plot_frame[plot_frame["ISO3"] == code].sort_values("Year")
        if country_rows.empty:
            continue
        is_kazakhstan = country == "Kazakhstan"
        color = highlight_color if is_kazakhstan else base_color
        linewidth = 2.8 if is_kazakhstan else 1.6
        alpha = 0.95 if is_kazakhstan else 0.65

        ax.plot(
            country_rows["Year"],
            country_rows["Total fertility rate"],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

        final_point = country_rows.iloc[-1]
        endpoint_y = float(final_point["Total fertility rate"])
        endpoint_x = float(final_point["Year"]) + 0.1
        label_offset = {"Tajikistan": 0.25, "Kyrgyzstan": -0.25}.get(country, 0.0)
        if country in east_europe_group:
            east_endpoints.append((endpoint_x, endpoint_y))
            continue
        ax.text(
            endpoint_x,
            endpoint_y + label_offset,
            country,
            color=color,
            fontsize=12,
            va="center",
            ha="left",
            weight="bold" if is_kazakhstan else "normal",
        )

    if east_endpoints:
        avg_x = max(x for x, _ in east_endpoints)
        avg_y = sum(y for _, y in east_endpoints) / len(east_endpoints)
        min_y = min(y for _, y in east_endpoints)
        max_y = max(y for _, y in east_endpoints)
        bracket_x = avg_x + 0.07
        ax.vlines(bracket_x, min_y, max_y, colors=base_color, linestyles="-", linewidth=1)
        ax.plot([bracket_x], [avg_y], marker="o", color=base_color, markersize=4)
        ax.text(
            bracket_x + 0.15,
            avg_y,
            "Eastern Europe",
            color=base_color,
            fontsize=12,
            va="center",
            ha="left",
        )

    if not plot_frame.empty:
        ax.set_xlim(plot_frame["Year"].min(), plot_frame["Year"].max() + 1.2)
        lower_bound = max(0, plot_frame["Total fertility rate"].min() - 0.2)
        upper_bound = max(plot_frame["Total fertility rate"].max() + 0.2, replacement_level + 0.2)
        ax.set_ylim(bottom=lower_bound, top=upper_bound)
    ax.set_ylabel("Births per woman")
    ax.set_xlabel("Year")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.axhline(replacement_level, color="#d62828", linestyle="--", linewidth=2)
    x_max = ax.get_xlim()[1]
    ax.text(
        x_max - 1.3,
        replacement_level + 0.05,
        "Replacement level (2.1)",
        color="#d62828",
        fontsize=12,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d62828", alpha=0.8),
    )
    _annotate_and_save(fig, "fig_tfr", "Figure: Total fertility rates by country, last decade")


def export_age_density(work: pd.DataFrame) -> None:
    age_subset = work[["aage", "religion"]].dropna()
    if age_subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("Set2", n_colors=age_subset["religion"].nunique())
    grouped = sorted(age_subset.groupby("religion"), key=lambda item: str(item[0]))
    for color, (group, group_data) in zip(palette, grouped):
        sns.kdeplot(
            data=group_data,
            x="aage",
            common_norm=False,
            fill=False,
            linewidth=2,
            color=color,
            ax=ax,
            label=str(group),
        )
    ax.set_xlim(left=max(15, age_subset["aage"].min()), right=min(80, age_subset["aage"].max()))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.legend(title="Group")
    _annotate_and_save(fig, "fig_age_density", "Figure: Age density by religious denomination")


def export_parity(work: pd.DataFrame) -> None:
    children_subset = work[["numbiol", "religion"]].dropna()
    if children_subset.empty:
        return
    parity_counts = (
        children_subset.groupby(["religion", "numbiol"])
        .size()
        .reset_index(name="respondents")
    )
    parity_counts["share"] = parity_counts.groupby("religion")["respondents"].transform(
        lambda counts: counts / counts.sum()
    )
    parities = sorted(parity_counts["numbiol"].unique())
    muslim_share = (
        parity_counts[parity_counts["religion"] == "Muslim"]
        .set_index("numbiol")["share"]
    )
    non_muslim_share = (
        parity_counts[parity_counts["religion"] == "Non-Muslim"]
        .set_index("numbiol")["share"]
    )
    muslim_values = [muslim_share.get(parity, 0) for parity in parities]
    non_muslim_values = [-non_muslim_share.get(parity, 0) for parity in parities]
    max_share = max(muslim_values + [-value for value in non_muslim_values], default=0) or 0.01
    colors = sns.color_palette("colorblind", 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(parities, non_muslim_values, color=colors[0], label="Non-Muslim")
    ax.barh(parities, muslim_values, color=colors[1], label="Muslim")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-max_share * 1.1, max_share * 1.1)
    ax.set_xlabel("Share of respondents")
    ax.set_ylabel("Number of biological children")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{abs(value)*100:.0f}%"))
    _annotate_and_save(fig, "fig_parity", "Figure: Completed fertility by denomination")


# --- Results helpers --------------------------------------------------------
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


def _get_row(table: pd.DataFrame, variable_key: str) -> pd.Series:
    mask = table["Variable"].str.strip().str.lower() == variable_key.lower()
    if not mask.any():
        return pd.Series(dtype=object)
    return table.loc[mask].iloc[0]


def _format_spec_label(column: str) -> Tuple[str, int]:
    match = math.nan
    try:
        import re
        m = re.match(r"\\((\\d+)\\)\\s*(.*)", column)
    except Exception:
        m = None
    if m:
        order = int(m.group(1))
        base = m.group(2).strip()
        friendly = base.replace("_", " ").title()
        return f"{order} – {friendly}", order
    return column, 0


def _build_effect_frame(
    table: pd.DataFrame,
    variable_map: Dict[str, str],
) -> pd.DataFrame:
    records = []
    columns = [col for col in table.columns if col not in {"Variable", "Context"}]
    for var_key, label in variable_map.items():
        row = _get_row(table, var_key)
        if row.empty:
            continue
        for column in columns:
            value = parse_numeric(row.get(column, ""))
            if value is None:
                continue
            spec_label, order = _format_spec_label(column)
            records.append(
                {
                    "Specification": spec_label,
                    "Order": order,
                    "Effect": value,
                    "Coefficient": label,
                }
            )
    return pd.DataFrame(records)


def _ratio_limits(values: Iterable[float], baseline: float | None) -> tuple[float, float]:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return (0, 1.2)
    vmin = float(series.min())
    vmax = float(series.max())
    if baseline is not None:
        vmin = min(vmin, baseline)
        vmax = max(vmax, baseline)
    span = vmax - vmin
    margin = 0.1 if span == 0 else span * 0.12
    lower = max(0, vmin - margin)
    upper = vmax + margin
    return lower, upper


def _plot_effect_horizontal(effect_df: pd.DataFrame, x_label: str, *, min_limit: float | None = None, max_limit: float | None = None) -> plt.Figure | None:
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
        subset = ordered[ordered["Coefficient"] == coeff].sort_values("Spec position")
        ax.plot(
            subset["Effect"],
            subset["Spec position"],
            marker="o",
            linewidth=2,
            color=color,
            label=coeff,
        )
    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Specification")
    lower, upper = _ratio_limits(ordered["Effect"], 1.0)
    if min_limit is not None:
        lower = min_limit
    if max_limit is not None:
        upper = max_limit
    ax.set_xlim(lower, upper)
    ax.legend(loc="upper left", frameon=False)
    return fig


def _plot_category_lollipops(effect_df: pd.DataFrame, x_label: str, *, upper_cap: float | None = None) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values("Level")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    positions = np.arange(len(ordered))
    ax.hlines(positions, 1.0, ordered["Effect"], color="#94a3b8", linewidth=2)
    ax.scatter(ordered["Effect"], positions, color="#1d4ed8", s=80, zorder=3, edgecolor="white", linewidth=0.8)
    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["Label"])
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    lower, upper = _ratio_limits(ordered["Effect"], 1.0)
    if upper_cap is not None:
        upper = upper_cap
    ax.set_xlim(0, upper)
    return fig


def _plot_birth_effects(effect_df: pd.DataFrame, *, upper_cap: float | None = None) -> plt.Figure | None:
    if effect_df.empty:
        return None
    ordered = effect_df.sort_values(["Order", "Coefficient"])
    categories = ordered.sort_values("Order")["Birth order"].unique()
    ordered["Birth order"] = pd.Categorical(ordered["Birth order"], categories=categories, ordered=True)
    unique_coeffs = ordered["Coefficient"].unique()
    palette = _build_palette(len(unique_coeffs))
    fig, ax = plt.subplots(figsize=(9, 4.5))
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
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="both"))
    lower, upper = _ratio_limits(ordered["Effect"], 1.0)
    if upper_cap is not None:
        upper = upper_cap
    ax.set_ylim(0, upper)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        legend.set_bbox_to_anchor((1.02, 1))
    return fig


def export_results_figures() -> None:
    # Figure: Muslim fertility advantage across specifications
    poisson_file = "Poisson_Muslim.xlsx"
    _, poisson_table = load_model_table(poisson_file)
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
    )
    fig_poisson = _plot_effect_horizontal(poisson_effects, "Incidence rate ratio", min_limit=0, max_limit=1.65)
    if fig_poisson:
        _annotate_and_save(fig_poisson, "fig_poisson_specs", "Figure: Muslim fertility advantage across specifications")

    # Figure: Value interactions (final spec, lollipop)
    values_file = "Poisson_Values.xlsx"
    _, values_table = load_model_table(values_file)
    interaction_value_effects = _build_effect_frame(
        values_table,
        {
            "1.muslim1#c.parents_support": "Parental support × Muslim",
            "1.muslim1#c.children_support": "Children support × Muslim",
            "1.muslim1#c.egalitarian": "Egalitarian × Muslim",
            "1.muslim1#c.family_support": "Family support × Muslim",
        },
    )
    interaction_value_effects = interaction_value_effects.assign(Level=np.arange(1, len(interaction_value_effects) + 1), Label=interaction_value_effects["Coefficient"])[["Level", "Label", "Effect"]]
    fig_interaction = _plot_category_lollipops(interaction_value_effects, "Incidence rate ratio", upper_cap=1.2)
    if fig_interaction:
        _annotate_and_save(fig_interaction, "fig_value_interactions", "Figure: Interaction effects (values × Muslim identification)")

    # Figure: Religiosity categories
    relig_file = "Poisson_relig.xlsx"
    _, relig_table = load_model_table(relig_file)
    religiosity_labels = {
        2: "Low religiosity",
        3: "Moderate religiosity",
        4: "High religiosity",
        5: "Very high religiosity",
    }
    category_records = []
    for level, label in religiosity_labels.items():
        row = _get_row(relig_table, f"recode of a1112 (religiousity) = {level}")
        if row.empty:
            continue
        value = parse_numeric(row.get("(1) Religiuos cat", ""))
        if value is None:
            continue
        category_records.append({"Level": level, "Label": f"{label} (cat. {level})", "Effect": value})
    category_df = pd.DataFrame(category_records)
    category_fig = _plot_category_lollipops(category_df, "Incidence rate ratio", upper_cap=1.2)
    if category_fig:
        _annotate_and_save(category_fig, "fig_religiosity_categories", "Figure: Incidence rate ratios by religiosity category")

    # Cox: Muslim households
    cox_muslim_file = "Cox_muslim.xlsx"
    _, cox_muslim_table = load_model_table(cox_muslim_file)
    birth_columns = [col for col in cox_muslim_table.columns if col.endswith("birth .")]
    cox_muslim_effects = _build_birth_effect_frame(cox_muslim_table, {"muslim1": "Muslim households"}, birth_columns)
    cap_candidate = cox_muslim_effects["Effect"].max() if not cox_muslim_effects.empty else None
    cox_upper = (max(cap_candidate, 1.0) + 0.2) if cap_candidate is not None and np.isfinite(cap_candidate) else None
    cox_muslim_fig = _plot_birth_effects(cox_muslim_effects, upper_cap=cox_upper)
    if cox_muslim_fig:
        _annotate_and_save(cox_muslim_fig, "fig_cox_muslim", "Figure: Hazard ratios for progressing births (Muslim vs. others)")

    # Cox: religiosity levels
    cox_rel_file = "Cox_rel.xlsx"
    _, cox_rel_table = load_model_table(cox_rel_file)
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
    base_cap = base_rel_effects["Effect"].max() if not base_rel_effects.empty else None
    base_upper = (max(base_cap, 1.0) + 0.2) if base_cap is not None and np.isfinite(base_cap) else None
    base_rel_fig = _plot_birth_effects(base_rel_effects, upper_cap=base_upper)
    if base_rel_fig:
        _annotate_and_save(base_rel_fig, "fig_cox_religiosity", "Figure: Birth timing by religiosity intensity")

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
    inter_cap = interaction_rel_effects["Effect"].max() if not interaction_rel_effects.empty else None
    inter_upper = (max(inter_cap, 1.0) + 0.2) if inter_cap is not None and np.isfinite(inter_cap) else None
    interaction_rel_fig = _plot_birth_effects(interaction_rel_effects, upper_cap=inter_upper)
    if interaction_rel_fig:
        _annotate_and_save(interaction_rel_fig, "fig_cox_interaction", "Figure: Combined effects of Muslim identity and religiosity on birth timing")


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
            import re

            match = re.match(r"(\\d+)_birth", column)
            if not match:
                continue
            number = int(match.group(1))
            records.append(
                {
                    "Birth order": f"{number}",
                    "Order": number,
                    "Effect": value,
                    "Coefficient": label,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    # Data & Methods
    export_sampling_chart()
    export_covariate_heatmap()

    # Descriptive statistics
    data_dir = Path("Data")
    export_tfr(data_dir / "post_soviet_tfr.csv")
    survey = load_kaz_ggs()
    work = survey.copy()
    if "numbiol" in work.columns:
        work["numbiol"] = pd.to_numeric(work["numbiol"], errors="coerce")
    if "aage" in work.columns:
        work["aage"] = pd.to_numeric(work["aage"], errors="coerce")
    religion = work.get("muslim")
    if religion is not None:
        work = work.assign(religion=religion.astype(str))
        work = work.replace({"religion": {"nan": np.nan}})
        export_age_density(work)
        export_parity(work)

    # Results
    export_results_figures()
    print(f"Saved figures to {EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Failed to export figures: {exc}", file=sys.stderr)
        sys.exit(1)
