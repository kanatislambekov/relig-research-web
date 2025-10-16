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

st.title("The role of religiosity and values in shaping fertility outcomes in Kazakhstan")  
st.info("Authors: Islambekov Kanat, Kozlov Vladimir, and Kazenin Konstantin")


st.header("Abstract")
st.markdown(
        """
    Kazakhstan represents a demographic outlier among post-Soviet states, experiencing a
        sustained fertility recovery since the late 1990s while most of the region stagnated at
        sub-replacement levels. This paper examines how religion, religiosity, and value orientations
        shape fertility outcomes in Kazakhstan using data from the 2018 Generations and Gender
        Survey (GGS). Poisson and Cox regression models show that Muslims consistently report
        more children than non-Muslims, even after controlling for socio-demographic
        characteristics. Religiosity has a non-linear effect as modest levels already raise fertility
        among Muslims, while religiosity has no effect for non-Muslims. Value orientations exert
        distinct influences: family-support norms increase fertility, especially for Muslims, whereas
        gender egalitarianism reduces fertility across groups. Therefore, these findings demonstrate
        that Kazakhstan’s demographic trajectory is sustained by a combination of Muslim identity,
        kinship-based family values, and selective adoption of modern norms."""
)
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
        pd.to_numeric(survey["totalchildren"], errors="coerce").mean()
        if "totalchildren" in survey.columns
        else None
    )

    median_age = (
    pd.to_numeric(survey["aage"], errors="coerce").median()
    if "aage" in survey.columns else None
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Respondents analysed", f"{total_cases:,}")
    with col2:
        st.metric("Share identifying as Muslim", f"{share_muslim * 100:.1f}%")
    with col3:
        if avg_children is not None:
            st.metric("Average total children", f"{avg_children:.2f}")
        else:
            st.metric("Average total children", "n/a")
    with col4:
        if median_age is not None:
            st.metric("Median age", f"{median_age:.1f} years")
        else:
            st.metric("Median age", "n/a")
else:
    st.info("Upload the Kaz_Ggs dataset to display key metrics from the survey.")

