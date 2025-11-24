"""Descriptive statistics derived from the Kazakh GGS dataset."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
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
COUNTRY_CODE_TO_NAME = {code: name for name, code in POST_SOVIET_COUNTRIES}
TFR_INDICATOR = "SP.DYN.TFRT.IN"
START_YEAR = 1991
END_YEAR = pd.Timestamp.now().year
POST_SOVIET_TFR_PATH = DATA_DIR / "post_soviet_tfr.csv"

sns.set_theme(style="darkgrid")

st.title("Descriptive statistics")
st.caption("Key descriptive patterns and national fertility trends")

# --- Total fertility rate trend ---------------------------------------------
st.header("Total Fertility Rates in post-Soviet states")


@st.cache_data(show_spinner=False)
def _load_post_soviet_tfr() -> pd.DataFrame:
    if POST_SOVIET_TFR_PATH.exists():
        return pd.read_csv(POST_SOVIET_TFR_PATH)

    def _download_post_soviet_tfr() -> pd.DataFrame:
        records: list[dict] = []
        for idx, (country, code) in enumerate(POST_SOVIET_COUNTRIES):
            params = {
                "format": "json",
                "per_page": 1000,
                "date": f"{START_YEAR}:{END_YEAR}",
            }
            url = f"https://api.worldbank.org/v2/country/{code}/indicator/{TFR_INDICATOR}"

            for attempt in range(3):
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 429 and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                payload = response.json()
                break
            else:
                raise ValueError(f"World Bank request failed repeatedly for {country}.")

            entries = payload[1] if len(payload) > 1 and payload[1] is not None else []
            for entry in entries:
                value = entry.get("value")
                year = entry.get("date")
                if value is None or year is None:
                    continue
                year_int = int(year)
                if year_int < START_YEAR:
                    continue
                records.append(
                    {
                        "Country": country,
                        "ISO3": code,
                        "Year": year_int,
                        "Total fertility rate": float(value),
                    }
                )

            # Gentle pacing to avoid additional rate limits.
            if idx < len(POST_SOVIET_COUNTRIES) - 1:
                time.sleep(0.3)

        if not records:
            raise ValueError("Fertility observations were empty after download.")

        frame = pd.DataFrame(records)
        POST_SOVIET_TFR_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(POST_SOVIET_TFR_PATH, index=False)
        return frame

    try:
        return _download_post_soviet_tfr()
    except (requests.RequestException, ValueError) as error:
        raise ValueError(
            "Unable to download World Bank fertility data automatically. "
            f"Download SP.DYN.TFRT.IN for post-Soviet countries (1991 onward) "
            f"and save it to {POST_SOVIET_TFR_PATH} with columns "
            "'Country', 'ISO3', 'Year', 'Total fertility rate'. "
            f"Original error: {error}"
        ) from error


try:
    tfr_series = _load_post_soviet_tfr()
except (requests.RequestException, ValueError) as error:
    st.warning(f"Unable to retrieve World Bank TFR data: {error}")
else:
    recent_years = tfr_series["Year"].unique()
    if recent_years.size > 0:
        max_year = int(recent_years.max())
        min_year = max(max_year - 10, int(tfr_series["Year"].min()))
        plot_frame = tfr_series[tfr_series["Year"].between(min_year, max_year)].copy()
    else:
        plot_frame = tfr_series.copy()

    fig_tfr, ax_tfr = plt.subplots(figsize=(9, 5))
    highlight_color = "#1b8a5a"
    baseline_color = "#b3b3b3"

    for country, code in POST_SOVIET_COUNTRIES:
        country_rows = plot_frame[plot_frame["ISO3"] == code].sort_values("Year")
        if country_rows.empty:
            continue
        is_kazakhstan = country == "Kazakhstan"
        color = highlight_color if is_kazakhstan else baseline_color
        linewidth = 2.5 if is_kazakhstan else 1.2
        alpha = 1.0 if is_kazakhstan else 0.8

        ax_tfr.plot(
            country_rows["Year"],
            country_rows["Total fertility rate"],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=country if is_kazakhstan else None,
        )

        final_point = country_rows.iloc[-1]
        ax_tfr.text(
            final_point["Year"] + 0.1,
            final_point["Total fertility rate"],
            country,
            color=color,
            fontsize=8,
            va="center",
        )

    if not plot_frame.empty:
        ax_tfr.set_xlim(plot_frame["Year"].min(), plot_frame["Year"].max() + 1.2)
        ax_tfr.set_ylim(
            bottom=max(0, plot_frame["Total fertility rate"].min() - 0.2),
            top=plot_frame["Total fertility rate"].max() + 0.2,
        )
    ax_tfr.set_ylabel("Births per woman")
    ax_tfr.set_xlabel("Year")
    ax_tfr.grid(True, axis="y", alpha=0.3, linestyle="--")
    if any(country == "Kazakhstan" for country, _ in POST_SOVIET_COUNTRIES):
        ax_tfr.legend(loc="upper left")
    st.pyplot(fig_tfr)
    plt.close(fig_tfr)

    st.markdown(
        """
        Kazakhstan remains among the higher-fertility post-Soviet states,
        exceeded only by its Central Asian neighbours. The Baltic states cluster
        near replacement level, while the South Caucasus has converged toward
        the European average over the past decade.
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

    fig_children, ax_children = plt.subplots(figsize=(8, 5))
    ax_children.barh(
        parities,
        non_muslim_values,
        color=colors[0],
        label="Non-Muslim",
    )
    ax_children.barh(
        parities,
        muslim_values,
        color=colors[1],
        label="Muslim",
    )
    ax_children.axvline(0, color="black", linewidth=1)
    ax_children.set_xlim(-max_share * 1.1, max_share * 1.1)
    ax_children.set_xlabel("Share of respondents")
    ax_children.set_ylabel("Number of biological children")
    ax_children.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    ax_children.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{abs(value)*100:.0f}%"))
    st.pyplot(fig_children)
    plt.close(fig_children)

    st.markdown(
        """
        The parity pyramid makes the contrast explicit: Muslim respondents are
        over-represented at three or more children, while non-Muslims cluster
        on the left side with one or two children.
        """
    )
