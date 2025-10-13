"""Interactive visualisations built from the Kaz_Ggs survey microdata."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data_loader import load_kaz_ggs


PALETTE = ["#4361EE", "#3A0CA3", "#F72585", "#4CC9F0", "#7209B7", "#B5179E", "#4895EF"]
HEATMAP_RANGE = ["#081C15", "#1B4332", "#40916C", "#74C69D", "#B7E4C7", "#D8F3DC"]


@st.cache_data(show_spinner=False)
def _typed_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return numeric_cols, categorical_cols


def _guess_column(columns: list[str], keywords: tuple[str, ...]) -> str | None:
    keywords_lower = tuple(keyword.lower() for keyword in keywords)
    for column in columns:
        name = column.lower()
        if any(keyword in name for keyword in keywords_lower):
            return column
    return columns[0] if columns else None


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for column in work.columns:
        if pd.api.types.is_numeric_dtype(work[column]):
            continue
        work[column] = work[column].astype(str).str.strip()
    return work


st.title("Kazakhstan Generations & Gender Survey visual narratives")

try:
    raw_data = load_kaz_ggs()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
except (ImportError, ValueError) as error:
    st.error(f"{error}")
    st.stop()

data = _clean_frame(raw_data)
numeric_columns, categorical_columns = _typed_columns(data)

if data.empty:
    st.warning("The Kaz_Ggs dataset did not contain any rows. Upload a populated file to explore the visuals.")
    st.stop()

if not numeric_columns:
    st.error("No numeric columns were detected. Confirm that the dataset includes measures such as age, fertility, or value indexes.")
    st.stop()

if not categorical_columns:
    st.error("No categorical columns were detected. Ensure that the dataset includes identifiers like religion or region.")
    st.stop()

st.markdown(
    """
These visuals highlight who the Kazakh respondents are and how fertility expectations relate to religiosity and values. 
Use the dropdowns to swap in alternative variables if your version of the survey names them differently.
    """
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Respondents", f"{len(data):,}")
with col_b:
    numeric_preview = _guess_column(numeric_columns, ("weight", "wgt"))
    if numeric_preview:
        st.metric("Average survey weight", f"{data[numeric_preview].mean():.2f}")
    else:
        st.metric("Average survey weight", "N/A")
with col_c:
    categorical_preview = _guess_column(categorical_columns, ("region", "oblast", "area"))
    if categorical_preview:
        st.metric("Regions represented", data[categorical_preview].nunique())
    else:
        st.metric("Regions represented", "N/A")

st.divider()

# --- Age distribution visual -------------------------------------------------
st.subheader("Age profile by faith tradition")

age_column_default = _guess_column(numeric_columns, ("age",))
religion_column_default = _guess_column(categorical_columns, ("relig", "faith", "denom", "muslim"))

if not age_column_default or not religion_column_default:
    st.info("Select the columns to plot the age distribution.")

age_column = st.selectbox("Numeric age column", options=numeric_columns, index=numeric_columns.index(age_column_default) if age_column_default in numeric_columns else 0)
religion_column = st.selectbox(
    "Religious identity column",
    options=categorical_columns,
    index=categorical_columns.index(religion_column_default) if religion_column_default in categorical_columns else 0,
)

bin_width = st.slider("Age bandwidth (years)", min_value=1, max_value=10, value=3, step=1)

age_frame = data[[age_column, religion_column]].dropna()

if age_frame.empty:
    st.warning("No rows contain both age and the selected religion field.")
else:
    density = (
        alt.Chart(age_frame)
        .transform_density(
            age_column,
            groupby=[religion_column],
            steps=200,
            bandwidth=bin_width,
            as_=[age_column, "density"],
        )
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X(f"{age_column}:Q", title="Age"),
            y=alt.Y("density:Q", title="Share of respondents"),
            color=alt.Color(f"{religion_column}:N", scale=alt.Scale(range=PALETTE)),
            tooltip=[religion_column, alt.Tooltip(f"{age_column}:Q", format=".1f"), alt.Tooltip("density:Q", format=".3f")],
        )
    )
    st.altair_chart(density, use_container_width=True)
    st.caption("Smoothed densities highlight how different religious groups skew younger or older.")

st.divider()

# --- Religiosity vs fertility -------------------------------------------------
st.subheader("Fertility expectations vs. religiosity intensity")

children_column_default = _guess_column(numeric_columns, ("child", "kids", "birth", "fert"))
religiosity_score_default = _guess_column(numeric_columns, ("relig", "attend", "pray", "belief", "piety", "faith"))
group_column_default = religion_column_default if religion_column_default else _guess_column(categorical_columns, ("gender", "sex"))

children_column = st.selectbox(
    "Number of children column",
    options=numeric_columns,
    index=numeric_columns.index(children_column_default) if children_column_default in numeric_columns else 0,
)
religiosity_column = st.selectbox(
    "Religiosity or practice score",
    options=numeric_columns,
    index=numeric_columns.index(religiosity_score_default) if religiosity_score_default in numeric_columns else 0,
)
group_column = st.selectbox(
    "Colour points by",
    options=categorical_columns,
    index=categorical_columns.index(group_column_default) if group_column_default in categorical_columns else 0,
)

filtered = data[[children_column, religiosity_column, group_column]].dropna()

if filtered.empty:
    st.warning("No rows include the selected religiosity and fertility columns.")
else:
    scatter = (
        alt.Chart(filtered)
        .mark_circle(size=70, opacity=0.6)
        .encode(
            x=alt.X(f"{religiosity_column}:Q", title=f"{religiosity_column} (standardised)", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{children_column}:Q", title=f"{children_column}"),
            color=alt.Color(f"{group_column}:N", scale=alt.Scale(range=PALETTE)),
            tooltip=[group_column, alt.Tooltip(f"{religiosity_column}:Q", format=".2f"), alt.Tooltip(f"{children_column}:Q", format=".2f")],
        )
    )

    trend = (
        alt.Chart(filtered)
        .transform_regression(religiosity_column, children_column, groupby=[group_column])
        .mark_line(size=3)
        .encode(
            x=alt.X(f"{religiosity_column}:Q", title=f"{religiosity_column}"),
            y=alt.Y(f"{children_column}:Q"),
            color=alt.Color(f"{group_column}:N", scale=alt.Scale(range=PALETTE)),
        )
    )

    st.altair_chart((scatter + trend).interactive(), use_container_width=True)
    st.caption("Loess trend lines capture whether more devout respondents expect larger families.")

st.divider()

# --- Values heatmap ----------------------------------------------------------
st.subheader("Value indexes across social groups")

value_column_default = _guess_column(numeric_columns, ("value", "index", "support", "attitude"))
row_group_default = religion_column_default if religion_column_default else _guess_column(categorical_columns, ("region", "oblast"))
column_group_default = _guess_column(categorical_columns, ("education", "cohort", "gender", "sex"))

value_column = st.selectbox(
    "Index or attitude score",
    options=numeric_columns,
    index=numeric_columns.index(value_column_default) if value_column_default in numeric_columns else 0,
)
row_group = st.selectbox(
    "Rows",
    options=categorical_columns,
    index=categorical_columns.index(row_group_default) if row_group_default in categorical_columns else 0,
)
column_group = st.selectbox(
    "Columns",
    options=categorical_columns,
    index=categorical_columns.index(column_group_default) if column_group_default in categorical_columns else 0,
)

heatmap_source = (
    data[[value_column, row_group, column_group]]
    .dropna()
    .groupby([row_group, column_group], as_index=False)[value_column]
    .mean()
)

if heatmap_source.empty:
    st.warning("No combinations of the selected categories contain valid index scores.")
else:
    heatmap = (
        alt.Chart(heatmap_source)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X(f"{column_group}:N", sort="-y"),
            y=alt.Y(f"{row_group}:N", sort="-x"),
            color=alt.Color(
                f"{value_column}:Q",
                scale=alt.Scale(range=HEATMAP_RANGE),
                title=f"Mean {value_column}",
            ),
            tooltip=[row_group, column_group, alt.Tooltip(f"{value_column}:Q", format=".2f")],
        )
    )
    st.altair_chart(heatmap, use_container_width=True)
    st.caption("Deeper greens indicate higher agreement with the selected value statement.")

st.info("Download the prepared dataset from the sidebar's `...` menu if you want to run additional offline analyses.")
