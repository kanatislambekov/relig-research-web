"""Page visualising Poisson regression tables."""
from __future__ import annotations

import io

import streamlit as st

from app_utils import load_excel_table, load_raw_excel

st.title("Poisson Regression Analysis")
st.write(
    "The Poisson models estimate the expected number of children across "
    "different specifications. Explore how the inclusion of controls, value "
    "indexes, and alternative religious categorizations change the results."
)

poisson_files = [
    (
        "Poisson_Muslim.xlsx",
        "Comparing Muslim and non-Muslim fertility with increasingly rich sets of controls.",
    ),
    (
        "Poisson_Values.xlsx",
        "Incorporating family-support and egalitarian value indexes alongside religious denomination.",
    ),
    (
        "Poisson_relig.xlsx",
        "Testing linear and categorical religiosity measures.",
    ),
]

for filename, description in poisson_files:
    title, table = load_excel_table(filename)

    st.subheader(title)
    st.caption(description)
    st.dataframe(table, use_container_width=True)

    raw_bytes = load_raw_excel(filename)
    st.download_button(
        label=f"Download {filename}",
        data=io.BytesIO(raw_bytes),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("---")

st.info(
    "Interpretation tip: rows without a variable name report the standard errors "
    "(in parentheses) corresponding to the coefficient listed immediately above."
)
