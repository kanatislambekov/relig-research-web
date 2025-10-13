"""Page focused on value-index regressions."""
from __future__ import annotations

import io

import streamlit as st

from app_utils import load_excel_table, load_raw_excel

st.title("Value Index Comparisons")
st.write(
    "This table summarises how value-oriented indexes relate to fertility when "
    "Muslim and non-Muslim respondents are modelled separately. Use the "
    "selector below to focus on specific model specifications."
)

title, table = load_excel_table("Indexes.xlsx")
available_models = [col for col in table.columns if col != "Variable"]

selected_models = st.multiselect(
    "Select model columns to display",
    options=available_models,
    default=available_models,
)

if selected_models:
    display_df = table[["Variable", *selected_models]]
else:
    st.warning("Please choose at least one model to display.")
    display_df = table[["Variable"]]

st.subheader(title)
st.dataframe(display_df, use_container_width=True)

raw_bytes = load_raw_excel("Indexes.xlsx")
st.download_button(
    label="Download Indexes.xlsx",
    data=io.BytesIO(raw_bytes),
    file_name="Indexes.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

st.info(
    "The models include egalitarian attitudes, parental support, and children's "
    "support indexes. Coefficients with asterisks are statistically significant "
    "at conventional levels."
)
