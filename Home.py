"""Landing page for the fertility and religiosity research site."""
from __future__ import annotations

from textwrap import dedent

import pandas as pd
import streamlit as st

from data_loader import load_docx_paragraphs, load_kaz_ggs

st.set_page_config(
    page_title="Fertility, Values, and Religion in Kazakhstan",
    page_icon="📊",
    layout="wide",
)

st.title("Fertility, Values, and Religion in Kazakhstan")
st.subheader("Exploring how belief systems shape family formation")

st.markdown(
    """
    This site summarises empirical findings from the Generations and Gender
    Survey in Kazakhstan. It connects descriptive statistics with regression
    models to explain how religiosity and value orientations shape fertility
    outcomes.
    """
)

st.header("Introduction")
try:
    paragraphs = load_docx_paragraphs("Text (1).docx")
except FileNotFoundError:
    paragraphs = []

if paragraphs:
    intro_text = "\n\n".join(paragraphs[:3])
else:
    intro_text = dedent(
        """
        The research investigates how religious affiliation and family-value
        orientations correlate with completed fertility and the timing of
        births. It integrates national survey data with regression models to
        evaluate whether value systems provide independent explanatory power
        beyond socio-demographic controls.
        """
    ).strip()

st.markdown(intro_text)

st.header("Research questions")
st.markdown(
    """
    1. How do fertility outcomes differ between Muslim and non-Muslim
       households once demographic and regional factors are held constant?
    2. To what extent do family-support and egalitarian value indexes explain
       variations in completed fertility and birth spacing?
    3. How does the intensity of religiosity interact with value orientations
       to influence the probability of progressing to higher-order births?
    """
)

st.header("Data snapshot")
try:
    survey = load_kaz_ggs()
except FileNotFoundError as error:
    st.warning(str(error))
    survey = None
except (ImportError, ValueError) as error:
    st.warning(f"{error}")
    survey = None

if survey is not None:
    total_cases = len(survey)
    share_muslim = survey["muslim"].value_counts(normalize=True).get("Muslim", 0.0)
    avg_children = (
        pd.to_numeric(survey["numbiol"], errors="coerce").mean()
        if "numbiol" in survey.columns
        else None
    )
    median_age = survey["aage"].median() if "aage" in survey.columns else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Respondents analysed", f"{total_cases:,}")
    with col2:
        st.metric("Share identifying as Muslim", f"{share_muslim * 100:.1f}%")
    with col3:
        if avg_children is not None:
            st.metric("Average biological children", f"{avg_children:.2f}")
        else:
            st.metric("Average biological children", "n/a")
    with col4:
        if median_age is not None:
            st.metric("Median age", f"{median_age:.1f} years")
        else:
            st.metric("Median age", "n/a")
else:
    st.info("Upload the Kaz_Ggs dataset to display key metrics from the survey.")

st.caption("Use the sidebar to navigate through descriptive statistics, modelling results, and interpretive conclusions.")
