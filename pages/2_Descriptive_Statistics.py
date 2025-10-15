"""Descriptive statistics derived from the Kazakh GGS dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
import plotly.express as px

from data_loader import load_kaz_ggs


DATA_DIR = Path("/Users/kiramaya/Documents/GitHub/relig-research-web/Data")
REGION_NAME_MAP = {
    "ABAY REGION": "Abai",
    "AKMOLA REGION": "Akmola",
    "AKTOBE REGION": "Aktobe",
    "ALMATY CITY": "Almaty (city)",
    "ALMATY REGION": "Almaty",
    "ASTANA CITY": "Astana",
    "ATYRAU REGION": "Atyrau",
    "BATYS-KAZAKHSTAN REGION": "West Kazakhstan",
    "KARAGANDY REGION": "Karaganda",
    "KOSTANAY REGION": "Kostanay",
    "KYZYLORDA REGION": "Kyzylorda",
    "MANGYSTAU REGION": "Mangystau",
    "PAVLODAR REGION": "Pavlodar",
    "SHYGYS KAZAKHSTAN REGION": "East Kazakhstan",
    "SHYMKENT CITY": "Shymkent (city)",
    "SOLTUSTIK KAZAKHSTAN REGION": "North Kazakhstan",
    "TURKISTAN REGION": "Turkestan",
    "ULYTAU REGION": "Ulytau",
    "ZHAMBYL REGION": "Jambyl",
    "ZHETISU REGION": "Jetisu",
}

sns.set_theme(style="whitegrid")

st.title("Descriptive statistics")
st.caption("Key descriptive patterns and national fertility trends")

# --- Total fertility rate trend ---------------------------------------------
st.header("Total Fertility Rate in Kazakhstan")
tfr_series = pd.DataFrame(
    {
        "Year": [2018, 2021, 2024],
        "Total fertility rate": [2.93, 3.35, 3.02],
    }
)

fig_tfr, ax_tfr = plt.subplots(figsize=(8, 4))
sns.lineplot(data=tfr_series, x="Year", y="Total fertility rate", marker="o", ax=ax_tfr)
ax_tfr.set_ylim(bottom=0, top=3.8)
ax_tfr.set_ylabel("Births per woman")
ax_tfr.grid(True, axis="y", alpha=0.3)
for _, row in tfr_series.iterrows():
    ax_tfr.annotate(f"{row['Total fertility rate']:.2f}", (row["Year"], row["Total fertility rate"] + 0.1), ha="center")
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


# --- Regional TFR map -------------------------------------------------------
st.header("Regional Total Fertility Rates")
@st.cache_data(show_spinner=False)
def _load_tfr_inputs() -> tuple[pd.DataFrame, dict]:
    csv_path = DATA_DIR / "TFR.csv"
    geojson_path = DATA_DIR / "kz.json"
    if not csv_path.exists():
        raise FileNotFoundError("The regional TFR CSV file is missing from the Data directory.")
    if not geojson_path.exists():
        raise FileNotFoundError("The Kazakhstan geojson file is missing from the Data directory.")

    tfr_raw = pd.read_csv(csv_path, dtype=str).rename(columns=lambda col: col.strip())
    tfr_raw["Region"] = tfr_raw["Region"].str.strip()

    year_columns = [column for column in tfr_raw.columns if column != "Region"]
    for column in year_columns:
        series = (
            tfr_raw[column]
            .astype(str)
            .str.replace("\u00a0", " ", regex=False)
            .str.replace("\u202f", " ", regex=False)
            .str.strip()
        )
        series = series.replace({"": np.nan, "nan": np.nan})
        series = series.str.replace(r"(?<=\d)\s+(?=\d)", ".", regex=True)
        tfr_raw[column] = pd.to_numeric(series, errors="coerce")

    tfr_raw["GeoName"] = tfr_raw["Region"].map(REGION_NAME_MAP)

    with geojson_path.open("r", encoding="utf-8") as source:
        geojson = json.load(source)

    return tfr_raw, geojson


try:
    regional_tfr, kazakhstan_geojson = _load_tfr_inputs()
except FileNotFoundError as error:
    st.warning(str(error))
else:
    missing_regions = regional_tfr[regional_tfr["GeoName"].isna()]["Region"].dropna().unique()
    if len(missing_regions) > 0:
        st.warning(
            "The following regions could not be matched to the geojson file: "
            + ", ".join(sorted(missing_regions))
        )

    available_years = [column for column in regional_tfr.columns if column not in {"Region", "GeoName"}]
    if available_years:
        available_years = sorted(available_years)
        selected_year = st.radio("Select year", options=available_years, index=len(available_years) - 1, horizontal=True)

        map_frame = (
            regional_tfr[["Region", "GeoName", selected_year]]
            .dropna(subset=["GeoName", selected_year])
            .rename(columns={selected_year: "Total fertility rate"})
        )

        color_range = (
            float(map_frame["Total fertility rate"].min()),
            float(map_frame["Total fertility rate"].max()),
        )

        fig_map = px.choropleth(
            map_frame,
            geojson=kazakhstan_geojson,
            locations="GeoName",
            featureidkey="properties.name",
            color="Total fertility rate",
            hover_name="Region",
            color_continuous_scale="Viridis",
            range_color=color_range,
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            margin={"r": 0, "t": 10, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Births per woman"),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown(
            """
            Regional total fertility rates reveal pronounced spatial contrasts. Western oil
            regions and the southern oblasts sustain the highest parity, while urban centres
            such as Astana and Almaty trail the national average.
            """
        )

# --- Load survey data --------------------------------------------------------
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
st.header("Age distribution by religious denomination")
age_subset = work[["aage", "religion"]].dropna()
if not age_subset.empty:
    fig_age, ax_age = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        data=age_subset,
        x="aage",
        fill=True,
        common_norm=False,
        color="#bdbdbd",
        alpha=0.35,
        linewidth=0,
        ax=ax_age,
        label="All respondents",
    )

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
            ax=ax_age,
            label=str(group),
        )

    ax_age.set_xlim(left=max(15, age_subset["aage"].min()), right=min(80, age_subset["aage"].max()))
    ax_age.set_ylim(bottom=0)
    ax_age.set_xlabel("Age")
    ax_age.set_ylabel("Density")
    ax_age.legend(title="Group")
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
st.header("Fertility by religious denomination")
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
    st.pyplot(fig_children)
    plt.close(fig_children)

    st.markdown(
        """
        Completed fertility remains higher within the Muslim population across
        the distribution, with thicker tails above three children. The
        difference persists even when focusing on the interquartile range.
        """
    )

