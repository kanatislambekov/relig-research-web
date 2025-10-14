"""Utility functions for loading research artefacts into Streamlit pages."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "Results"
SURVEY_DIR = BASE_DIR / "data"


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected data file {path} to exist")
    return path


def _read_excel_raw(file_name: str) -> pd.DataFrame:
    path = _ensure_exists(RESULTS_DIR / file_name)
    df = pd.read_excel(path, header=None)
    # Normalise missing values so we can easily filter empty structural rows.
    df = df.fillna("")
    mask = ~df.apply(lambda row: row.astype(str).str.strip().eq("").all(), axis=1)
    df = df[mask].reset_index(drop=True)
    return df


def _flatten_headers(upper: Iterable, lower: Iterable) -> List[str]:
    columns: List[str] = []
    for top, bottom in zip(upper, lower):
        parts = []
        for candidate in (top, bottom):
            if isinstance(candidate, str):
                candidate = candidate.strip()
            if candidate and str(candidate).strip().lower() != "nan":
                parts.append(str(candidate).strip())
        columns.append(" ".join(parts) if parts else "Metric")
    return columns


def _tidy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw regression table into a viewer-friendly structure."""
    work = df.copy()
    first_column = work.columns[0]
    records = []
    context = ""
    last_variable = ""
    summary_keywords = {"observations", "log likelihood", "pseudo r-squared", "wald chi2", "lr chi2"}

    for _, row in work.iterrows():
        variable_cell = str(row[first_column]).strip()
        metrics = row.drop(first_column)
        metrics_filled = metrics.fillna("").astype(str).str.strip()
        metrics_blank = metrics_filled.eq("").all()

        if metrics_blank:
            if variable_cell:
                context = variable_cell
            continue

        if not variable_cell:
            display_name = f"{last_variable} (Std. Err.)" if last_variable else "Std. Err."
        else:
            display_name = variable_cell
            last_variable = variable_cell

        display_context = ""
        if context and display_name.strip().lower() not in summary_keywords and context != display_name:
            display_context = context

        record = {"Variable": display_name, "Context": display_context}
        for column_name, value in metrics.items():
            if isinstance(value, float) and pd.isna(value):
                record[str(column_name)] = ""
            else:
                record[str(column_name)] = value
        records.append(record)

    tidy_df = pd.DataFrame(records).fillna("")
    return tidy_df


@st.cache_data(show_spinner=False)
def load_model_table(file_name: str) -> Tuple[str, pd.DataFrame]:
    """Load and tidy one of the regression result tables."""
    raw = _read_excel_raw(file_name)
    if raw.empty:
        return "", pd.DataFrame()

    title = str(raw.iloc[0, 0]).strip()
    if len(raw) < 3:
        return title, raw

    header_upper = raw.iloc[1]
    header_lower = raw.iloc[2]
    columns = _flatten_headers(header_upper, header_lower)
    body = raw.iloc[3:].reset_index(drop=True)
    body.columns = columns

    tidy = _tidy_table(body)
    return title, tidy


@st.cache_data(show_spinner=False)
def load_docx_paragraphs(file_name: str) -> List[str]:
    """Extract plain-text paragraphs from a DOCX document."""
    path = _ensure_exists(RESULTS_DIR / file_name)
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        text = "".join(texts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def parse_numeric(value: str | float | int) -> float | None:
    """Extract the numeric portion of a regression cell, ignoring significance stars."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    cleaned = cleaned.replace("(", "").replace(")", "")
    match = re.match(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def count_stars(value: str | float | int) -> int:
    """Return the number of significance stars present in a cell."""
    if not isinstance(value, str):
        return 0
    return value.count("*")


def _find_dataset(base_name: str) -> Path:
    """Return the path to a dataset that starts with ``base_name`` inside ``SURVEY_DIR``."""

    if not SURVEY_DIR.exists():
        raise FileNotFoundError(
            "The survey data folder was not found. Place the Kaz_Ggs dataset inside a `data/` directory at the "
            "project root."
        )

    normalised_base = base_name.lower()
    candidates = []
    for path in SURVEY_DIR.iterdir():
        if not path.is_file():
            continue
        if path.stem.lower() == normalised_base or path.name.lower().startswith(normalised_base):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No dataset matching the pattern 'Kaz_Ggs.*' was found in the data folder. Confirm the filename "
            "and extension."
        )

    # Prefer modern column-preserving formats when multiple versions are present.
    priority = [
        ".parquet",
        ".feather",
        ".csv",
        ".xlsx",
        ".sav",
        ".dta",
    ]
    candidates.sort(key=lambda path: (priority.index(path.suffix.lower()) if path.suffix.lower() in priority else len(priority), path.name))
    return candidates[0]


def _read_spss(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Reading SPSS files requires the `pyreadstat` package. Add it to your environment to load .sav files."
        ) from exc

    df, _ = pyreadstat.read_sav(path)
    return df


def _read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Reading parquet files requires `pyarrow` or `fastparquet`. Install one of them to continue."
            ) from exc
    if suffix == ".feather":
        try:
            return pd.read_feather(path)
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Reading feather files requires `pyarrow`. Install it to continue.") from exc
    if suffix == ".sav":
        return _read_spss(path)
    if suffix == ".dta":
        return pd.read_stata(path)

    raise ValueError(f"Unsupported dataset extension: {suffix}")


@st.cache_data(show_spinner=False)
def load_kaz_ggs() -> pd.DataFrame:
    """Load the main Kazakh GGS survey dataset from the data folder."""

    path = _find_dataset("Kaz_Ggs")
    df = _read_dataset(path)

    if df.empty:
        raise ValueError("The Kaz_Ggs dataset appears to be empty. Verify that it contains rows.")

    # Normalise column names to strings so Streamlit widgets can display them safely.
    df.columns = [str(column).strip() for column in df.columns]
    return df
