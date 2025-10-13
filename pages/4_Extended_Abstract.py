"""Page rendering the extended abstract narrative."""
from __future__ import annotations

import streamlit as st

from app_utils.content import EXTENDED_ABSTRACT

st.title("Extended Abstract")
st.write(
    "The narrative below summarises the main insights from the accompanying "
    "paper. Each section condenses the results showcased on the other pages of "
    "the site."
)

for section in EXTENDED_ABSTRACT:
    st.subheader(section.title)
    for paragraph in section.paragraphs:
        st.write(paragraph)

st.markdown("---")
st.caption("Prepared for the data visualisation class project on religiosity and fertility in Kazakhstan.")
