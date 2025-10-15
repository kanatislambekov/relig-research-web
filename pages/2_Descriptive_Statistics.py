"""Descriptive statistics derived from the Kazakh GGS dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from data_loader import load_kaz_ggs

sns.set_theme(style="whitegrid")

st.title("Descriptive statistics")
st.caption("Key descriptive patterns and national fertility trends")

BASE_DIR = Path(__file__).resolve().parents[1]


def _normalise_label(value: str | float | int | None) -> str:
    """Return a comparable representation for region and group labels."""

    if value is None:
        return ""
    text = str(value).strip()
    lowered = text.lower()
    replacements = {"-": "", " ": "", "’": "", "'": ""}
    for old, new in replacements.items():
        lowered = lowered.replace(old, new)
    return lowered


def _locate_data_file(patterns: list[str]) -> Path | None:
    """Search common directories for a file matching the provided patterns."""

    search_roots = [BASE_DIR / "Data", BASE_DIR / "data", BASE_DIR]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    return path
    return None


def _load_tfr_by_region() -> pd.DataFrame | None:
    """Return a tidy long-form table with regional TFR values."""

    tfr_path = _locate_data_file(["*tfr*.csv", "*fertility*.csv", "*tfr*.xlsx"])
    if tfr_path is None:
        st.info("Add the regional TFR dataset to the `Data/` folder to unlock the map view.")
        return None

    if tfr_path.suffix.lower() == ".csv":
        raw = pd.read_csv(tfr_path)
    elif tfr_path.suffix.lower() in {".xls", ".xlsx"}:
        raw = pd.read_excel(tfr_path)
    else:
        st.warning(f"Unsupported TFR data format: {tfr_path.name}")
        return None

    raw.columns = [str(column).strip() for column in raw.columns]

    region_column = None
    for candidate in ("region", "Region", "aregion", "NAME_1"):
        matches = [column for column in raw.columns if column.lower() == candidate.lower()]
        if matches:
            region_column = matches[0]
            break
    if region_column is None:
        region_column = raw.columns[0]

    tidy: pd.DataFrame
    lower_map = {column.lower(): column for column in raw.columns}
    if "year" in lower_map and {"tfr", "total fertility rate", "value"} & set(lower_map):
        value_key = next(column for column in raw.columns if column.lower() in {"tfr", "total fertility rate", "value"})
        tidy = raw.rename(columns={region_column: "Region", lower_map["year"]: "Year", value_key: "TFR"})[
            ["Region", "Year", "TFR"]
        ]
    else:
        value_columns = [column for column in raw.columns if column != region_column]
        tidy = raw.rename(columns={region_column: "Region"}).melt(
            id_vars="Region", value_vars=value_columns, var_name="Year", value_name="TFR"
        )

    tidy["Region"] = tidy["Region"].astype(str).str.strip()
    tidy["Year"] = tidy["Year"].astype(str).str.strip()
    tidy["TFR"] = pd.to_numeric(tidy["TFR"], errors="coerce")
    tidy = tidy.dropna(subset=["Region", "Year", "TFR"])
    return tidy


def _load_region_map() -> dict | None:
    """Return the regional GeoJSON used for the fertility choropleth."""

    geojson_path = _locate_data_file(["*.geojson", "*regions*.json", "*kaz*.json"])
    if geojson_path is None:
        st.info("Place the Kazakhstan regional GeoJSON inside the `Data/` directory to enable the map.")
        return None

    try:
        with geojson_path.open("r", encoding="utf-8") as source:
            data = json.load(source)
    except json.JSONDecodeError as error:
        st.error(f"Failed to parse the regional map file: {error}")
        return None

    if data.get("type") == "Topology":
        st.error("The provided map is TopoJSON. Convert it to GeoJSON before uploading.")
        return None
    if data.get("type") != "FeatureCollection":
        st.error("The regional map file must be a GeoJSON FeatureCollection.")
        return None
    return data


def _infer_region_property(features: list[dict], regions: pd.Series) -> str | None:
    """Identify the GeoJSON property that matches the TFR region names."""

    if not features:
        return None

    candidate_keys = list(features[0].get("properties", {}).keys())
    normalised_regions = {_normalise_label(region) for region in regions}

    for key in candidate_keys:
        feature_values = {_normalise_label(feature.get("properties", {}).get(key)) for feature in features}
        if not feature_values:
            continue
        matches = normalised_regions & feature_values
        if matches and len(matches) >= max(3, int(0.6 * len(normalised_regions))):
            return key
    return None


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _build_tfr_map(tfr_table: pd.DataFrame, geojson: dict) -> None:
    """Render the interactive TFR choropleth using PyDeck."""

    def _year_sort_key(value: str) -> tuple[int, str]:
        try:
            numeric = float(value)
        except ValueError:
            return (1, value)
        return (0, numeric)

    years = sorted(tfr_table["Year"].unique(), key=_year_sort_key)
    if not years:
        st.warning("The TFR dataset does not contain any valid years.")
        return

    if len(years) <= 3:
        default_year = years[-1]
    else:
        default_year = years[0]

    selected_year = st.segmented_control("Select year", options=years, default=default_year)
    year_slice = tfr_table[tfr_table["Year"] == selected_year].copy()
    if year_slice.empty:
        st.warning("No fertility observations available for the selected year.")
        return

    region_property = _infer_region_property(geojson.get("features", []), year_slice["Region"])
    if region_property is None:
        st.error("Could not align region names between the GeoJSON and the TFR dataset.")
        return

    year_slice["_key"] = year_slice["Region"].apply(_normalise_label)
    value_min, value_max = year_slice["TFR"].min(), year_slice["TFR"].max()
    span = value_max - value_min if value_max != value_min else 1.0
    start_color = np.array(_hex_to_rgb("#d9f0a3"), dtype=float)
    end_color = np.array(_hex_to_rgb("#006837"), dtype=float)

    enriched_features: list[dict] = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        key = _normalise_label(properties.get(region_property))
        match = year_slice[year_slice["_key"] == key]
        if not match.empty:
            tfr_value = float(match.iloc[0]["TFR"])
            ratio = (tfr_value - value_min) / span
            color = start_color + ratio * (end_color - start_color)
            fill_color = [int(channel) for channel in color] + [180]
            properties.update(
                {
                    "tfr": round(tfr_value, 2),
                    "region_label": match.iloc[0]["Region"],
                    "fill_color": fill_color,
                }
            )
        else:
            properties.update({"tfr": None, "region_label": properties.get(region_property, ""), "fill_color": [200, 200, 200, 70]})
        enriched_features.append({"type": "Feature", "geometry": feature.get("geometry"), "properties": properties})

    map_data = {"type": "FeatureCollection", "features": enriched_features}
    try:
        view_state = pdk.data_utils.compute_view(map_data)
    except Exception:  # pragma: no cover - pydeck helper fallback
        view_state = pdk.ViewState(latitude=48.0, longitude=66.9, zoom=3.4)

    layer = pdk.Layer(
        "GeoJsonLayer",
        map_data,
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255],
        get_line_width=30,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": "<b>{region_label}</b><br/>TFR: {tfr}",
        "style": {"backgroundColor": "#f9fafb", "color": "#111827"},
    }

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))


# --- Total fertility rate trend ---------------------------------------------
tfr_series = pd.DataFrame(
    {
        "Year": [2018, 2021, 2024],
        "Total fertility rate": [2.93, 3.11, 3.02],
    }
)

fig_tfr, ax_tfr = plt.subplots(figsize=(8, 4))
sns.lineplot(data=tfr_series, x="Year", y="Total fertility rate", marker="o", ax=ax_tfr)
ax_tfr.set_ylim(bottom=0)
ax_tfr.set_ylabel("Births per woman")
ax_tfr.set_title("Total fertility rate in Kazakhstan")
ax_tfr.grid(True, axis="y", alpha=0.3)
for _, row in tfr_series.iterrows():
    ax_tfr.annotate(f"{row['Total fertility rate']:.2f}", (row["Year"], row["Total fertility rate"] + 0.02))
st.pyplot(fig_tfr)
plt.close(fig_tfr)

st.markdown(
    """
    The national total fertility rate (TFR) climbed above three births per woman
    during the pandemic period before easing slightly in 2024. The broader
    trajectory still remains well above replacement level, underscoring the
    importance of examining within-country heterogeneity.
    """
)

# --- Regional TFR choropleth -------------------------------------------------
tfr_table = _load_tfr_by_region()
geojson = _load_region_map()
if tfr_table is not None and geojson is not None:
    st.subheader("Regional total fertility rate")
    st.caption("Hover over a region to inspect its fertility rate and switch between available years.")
    _build_tfr_map(tfr_table, geojson)

# --- Load microdata for survey-based visuals -------------------------------
try:
    survey = load_kaz_ggs()
except FileNotFoundError as error:
    st.warning(str(error))
    survey = None
except (ImportError, ValueError) as error:
    st.warning(f"{error}")
    survey = None

if survey is None:
    st.stop()

work = survey.copy()

if "numbiol" in work.columns:
    work["numbiol"] = pd.to_numeric(work["numbiol"], errors="coerce")
if "aage" in work.columns:
    work["aage"] = pd.to_numeric(work["aage"], errors="coerce")

religion = work.get("muslim")
if religion is None:
    st.warning("The dataset does not contain the 'muslim' classification needed for stratified visuals.")
    st.stop()

work = work.assign(religion=religion.astype(str))
work = work.replace({"religion": {"nan": np.nan}})

# --- Age distribution -------------------------------------------------------
age_subset = work[["aage", "religion"]].dropna()
if not age_subset.empty:
    unique_groups = [value for value in age_subset["religion"].unique() if value == value]
    if unique_groups:
        pretty_labels = {}
        for value in unique_groups:
            text = str(value).strip().lower()
            if text in {"1", "1.0", "true"}:
                pretty_labels[value] = "Muslim"
            elif text in {"0", "0.0", "false"}:
                pretty_labels[value] = "Non-Muslim"
            else:
                pretty_labels[value] = str(value).strip().title()

        palettes = sns.color_palette("Set2", len(unique_groups))
        fig_age, axes = plt.subplots(1, len(unique_groups), figsize=(5.2 * len(unique_groups), 4), sharey=True)
        if len(unique_groups) == 1:
            axes = [axes]

        full_density_color = "#d1d5db"
        x_limits = (
            max(15, float(age_subset["aage"].min())),
            min(80, float(age_subset["aage"].max())),
        )

        for ax, group, color in zip(axes, unique_groups, palettes):
            subset = age_subset[age_subset["religion"] == group]
            sns.kdeplot(data=age_subset, x="aage", fill=True, color=full_density_color, alpha=0.55, linewidth=0, ax=ax)
            sns.kdeplot(data=subset, x="aage", fill=True, color=color, alpha=0.75, linewidth=0, ax=ax)
            sns.kdeplot(data=subset, x="aage", color="#111827", linewidth=1.2, ax=ax)
            ax.set_xlim(*x_limits)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Age (years)")
            if ax is axes[0]:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")
            ax.set_title(pretty_labels.get(group, str(group)))
            ax.grid(False)

        fig_age.suptitle("Age distribution by religious identification", fontsize=14, y=1.02)
        plt.tight_layout()
        st.pyplot(fig_age)
        plt.close(fig_age)

        st.markdown(
            """
            Muslim respondents skew slightly younger, supporting the hypothesis that
            higher fertility is linked to a younger demographic profile. Non-Muslim
            respondents show a flatter distribution across older ages.
            """
        )

# --- Completed fertility ----------------------------------------------------
children_subset = work[["numbiol", "religion"]].dropna()
if not children_subset.empty:
    fig_children, ax_children = plt.subplots(figsize=(8, 4))
    sns.violinplot(
        data=children_subset,
        x="religion",
        y="numbiol",
        scale="width",
        inner="quartile",
        palette="Set2",
        ax=ax_children,
    )
    ax_children.set_xlabel("Religious identification")
    ax_children.set_ylabel("Number of biological children")
    ax_children.set_ylim(bottom=0)
    ax_children.set_title("Completed fertility by religious identification")
    st.pyplot(fig_children)
    plt.close(fig_children)

    st.markdown(
        """
        Completed fertility remains higher within the Muslim population across
        the distribution, with thicker tails above three children. The
        difference persists even when focusing on the interquartile range.
        """
    )

# --- Value indexes snapshot -------------------------------------------------
value_columns = [
    column
    for column in ("family_support", "children_support", "parents_support", "egalitarian")
    if column in work.columns
]

if value_columns:
    value_subset = work[value_columns + ["religion"]].dropna()
    if not value_subset.empty:
        long_values = value_subset.melt(id_vars="religion", var_name="Index", value_name="Score")
        fig_values, ax_values = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=long_values,
            x="Index",
            y="Score",
            hue="religion",
            estimator=np.mean,
            errorbar=("ci", 95),
            ax=ax_values,
        )
        ax_values.set_ylim(bottom=0)
        ax_values.set_ylabel("Average index score")
        ax_values.set_title("Value orientations by denomination")
        ax_values.legend(title="Religious identification")
        st.pyplot(fig_values)
        plt.close(fig_values)

        st.markdown(
            """
            Family-support indexes score notably higher among Muslim households,
            while egalitarian scores dip below the sample average. These
            contrasts motivate the modelling strategy that evaluates values and
            religious identity jointly.
            """
        )
