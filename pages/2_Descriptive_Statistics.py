"""Descriptive statistics derived from the Kazakh GGS dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from data_loader import load_kaz_ggs

sns.set_theme(style="whitegrid")

st.title("Descriptive statistics")
st.caption("Key descriptive patterns and national fertility trends")

# --- Total fertility rate trend ---------------------------------------------
tfr_series = pd.DataFrame(
    {
        "Year": [2018, 2021, 2024],
        "Total fertility rate": [2.93, 3.35, 3.02],
    }
)

fig_tfr, ax_tfr = plt.subplots(figsize=(8, 4))
sns.lineplot(data=tfr_series, x="Year", y="Total fertility rate", marker="o", ax=ax_tfr)
ax_tfr.set_ylim(bottom=0, top=3.8)
ax_tfr.set_ylabel("Births per woman")
ax_tfr.set_title("Total fertility rate in Kazakhstan")
ax_tfr.grid(True, axis="y", alpha=0.3)
for _, row in tfr_series.iterrows():
    ax_tfr.annotate(f"{row['Total fertility rate']:.2f}", (row["Year"], row["Total fertility rate"] + 0.05))
st.pyplot(fig_tfr)
plt.close(fig_tfr)

st.markdown(
    """
    The national total fertility rate (TFR) climbed above three births per woman
    during the pandemic period before easing slightly in 2024. The broader
    trajectory still remains well above replacement level, underscoring the
    importance of examining within-country heterogeneity.
    """
)

# --- Load survey data --------------------------------------------------------
try:
    survey = load_kaz_ggs()
except FileNotFoundError as error:
    st.warning(str(error))
    survey = None
except (ImportError, ValueError) as error:
    st.warning(f"{error}")
    survey = None

if survey is None:
    st.stop()

work = survey.copy()

if "numbiol" in work.columns:
    work["numbiol"] = pd.to_numeric(work["numbiol"], errors="coerce")
if "aage" in work.columns:
    work["aage"] = pd.to_numeric(work["aage"], errors="coerce")

religion = work.get("muslim")
if religion is None:
    st.warning("The dataset does not contain the 'muslim' classification needed for stratified visuals.")
    st.stop()

work = work.assign(religion=religion.astype(str))
work = work.replace({"religion": {"nan": np.nan}})

# --- Age distribution -------------------------------------------------------
age_subset = work[["aage", "religion"]].dropna()
if not age_subset.empty:
    fig_age, ax_age = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        data=age_subset,
        x="aage",
        hue="religion",
        fill=True,
        common_norm=False,
        alpha=0.35,
        ax=ax_age,
    )
    ax_age.set_xlim(left=max(15, age_subset["aage"].min()), right=min(80, age_subset["aage"].max()))
    ax_age.set_ylim(bottom=0)
    ax_age.set_xlabel("Age")
    ax_age.set_ylabel("Density")
    ax_age.set_title("Age distribution by religious identification")
    st.pyplot(fig_age)
    plt.close(fig_age)

    st.markdown(
        """
        Muslim respondents skew slightly younger, supporting the hypothesis that
        higher fertility is linked to a younger demographic profile. Non-Muslim
        respondents show a flatter distribution across older ages.
        """
    )

# --- Completed fertility ----------------------------------------------------
children_subset = work[["numbiol", "religion"]].dropna()
if not children_subset.empty:
    fig_children, ax_children = plt.subplots(figsize=(8, 4))
    sns.violinplot(
        data=children_subset,
        x="religion",
        y="numbiol",
        scale="width",
        inner="quartile",
        palette="Set2",
        ax=ax_children,
    )
    ax_children.set_xlabel("Religious identification")
    ax_children.set_ylabel("Number of biological children")
    ax_children.set_ylim(bottom=0)
    ax_children.set_title("Completed fertility by religious identification")
    st.pyplot(fig_children)
    plt.close(fig_children)

    st.markdown(
        """
        Completed fertility remains higher within the Muslim population across
        the distribution, with thicker tails above three children. The
        difference persists even when focusing on the interquartile range.
        """
    )

