"""Page displaying Cox proportional hazards models."""
from __future__ import annotations

import io

import streamlit as st

from app_utils import load_excel_table, load_raw_excel

st.title("Birth-Order Timing (Cox Models)")
st.write(
    "The Cox proportional hazards models evaluate how quickly respondents reach "
    "each birth order. Hazard ratios above one indicate a higher likelihood of "
    "progressing to the specified birth relative to the baseline group."
)

cox_files = [
    (
        "Cox_muslim.xlsx",
        "Comparing Muslim and non-Muslim respondents across first to fourth births.",
    ),
    (
        "Cox_rel.xlsx",
        "Introducing religiosity measures alongside denomination to assess timing differences.",
    ),
]

for filename, description in cox_files:
    title, table = load_excel_table(filename)

    st.subheader(title if title.strip() else filename.replace("_", " "))
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
    "Remember: in hazard models, a value of 1.00 implies no difference with the "
    "reference group. Values above (below) 1 indicate faster (slower) progression "
    "to the next birth."
)
