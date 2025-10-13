"""Landing page for the research results Streamlit site."""
from __future__ import annotations

import streamlit as st

from app_utils import load_excel_table
from app_utils.content import KEY_FINDINGS

st.set_page_config(
    page_title="Fertility, Values, and Religion in Kazakhstan",
    layout="wide",
    page_icon="📊",
)

st.title("Fertility, Values, and Religion in Kazakhstan")
st.subheader("Exploring how religiosity and family values shape fertility outcomes")

intro = (
    "This interactive site accompanies the data visualisation class project and "
    "extended abstract on fertility in Kazakhstan. Use the navigation menu to "
    "explore regression tables, value indexes, and detailed narrative findings "
    "from the study."
)
st.markdown(intro)

# High-level metrics calculated from the uploaded Excel tables.
poisson_muslim_title, poisson_muslim = load_excel_table("Poisson_Muslim.xlsx")
poisson_values_title, poisson_values = load_excel_table("Poisson_Values.xlsx")
cox_title, cox = load_excel_table("Cox_muslim.xlsx")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Poisson specifications",
        f"{poisson_muslim.shape[1] - 1 + poisson_values.shape[1] - 1}",
        "Models comparing Muslim and non-Muslim fertility",
    )
with col2:
    st.metric(
        "Value indexes",
        "4",
        "Family support and egalitarian measures captured in the survey",
    )
with col3:
    st.metric(
        "Birth-order models",
        f"{cox.shape[1] - 1}",
        "Cox regressions for first through fourth births",
    )

st.markdown("---")
st.header("Key Findings")
st.write(
    "The extended abstract distils the statistical output into four central "
    "takeaways:"
)
st.markdown("\n".join(f"- {finding}" for finding in KEY_FINDINGS))

st.markdown("---")
st.header("How to Navigate")
st.write(
    "Select a page from the sidebar to dig deeper into specific analyses. Each "
    "page includes download buttons so you can reuse the exact tables that feed "
    "the visuals."
)

st.info(
    "Tip: Open the app in full-screen mode or collapse the sidebar to focus on the "
    "tables when presenting your results in class."
)

st.caption(
    "Data sources: regression output and indexes prepared for the class project."
)
