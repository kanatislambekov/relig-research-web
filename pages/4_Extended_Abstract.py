from pathlib import Path

import streamlit as st

from data_loader import load_docx_paragraphs

DOC_FILENAME = "Text (1).docx"
DOC_PATH = Path("Results") / DOC_FILENAME

st.title("Extended abstract")

paragraphs = load_docx_paragraphs(DOC_FILENAME)

if not paragraphs:
    st.warning("Extended abstract content could not be parsed. Ensure the DOCX file is stored in the Results folder.")
else:
    for paragraph in paragraphs:
        st.markdown(paragraph)

if DOC_PATH.exists():
    st.download_button(
        label="Download the original extended abstract",
        data=DOC_PATH.read_bytes(),
        file_name=DOC_FILENAME,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
else:
    st.info("The source document is not currently available for download.")
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
