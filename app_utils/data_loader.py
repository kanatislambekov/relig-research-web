"""Utility helpers for loading research result tables."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

# Resolve the repository root and data directory lazily so Streamlit works when
# executed from any page module.
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "Results"


def _resolve_results_path(filename: str) -> Path:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find {filename!r} in {RESULTS_DIR}")
    return path


@st.cache_data(show_spinner=False)
def load_excel_table(filename: str) -> Tuple[str, pd.DataFrame]:
    """Load an Excel table exported from regression output.

    The files share a common structure where the first row stores the table
    title, the second row stores column labels, and the following rows contain
    the body of the table (coefficients and standard errors).  This helper
    extracts that structure and returns a tidy :class:`~pandas.DataFrame` that
    Streamlit can render easily.
    """
    path = _resolve_results_path(filename)
    raw = pd.read_excel(path, header=None)

    title = str(raw.iat[0, 0]) if not raw.empty else filename
    header_row = raw.iloc[1].fillna("") if len(raw) > 1 else pd.Series(dtype=str)

    columns = []
    for idx, value in enumerate(header_row):
        if idx == 0:
            columns.append("Variable")
            continue
        label = str(value).strip()
        if not label:
            label = f"Model {idx}"
        columns.append(label)

    body = raw.iloc[2:].copy()
    if columns:
        body.columns = columns
    body = body.dropna(how="all")
    body = body.fillna("")

    return title, body


@st.cache_data(show_spinner=False)
def load_raw_excel(filename: str) -> bytes:
    """Return the raw bytes of an Excel file for download buttons."""
    return _resolve_results_path(filename).read_bytes()
