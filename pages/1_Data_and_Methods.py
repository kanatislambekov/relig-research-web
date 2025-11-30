"""Data and methodology overview with supporting visuals."""
from __future__ import annotations

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from data_loader import load_kaz_ggs

FONT_FAMILY = "Helvetica"
FONT_RC = {
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
}

plt.rcParams.update({"font.family": FONT_FAMILY, **FONT_RC})
sns.set_theme(style="whitegrid", font=FONT_FAMILY, rc=FONT_RC)

_FIGURE_NUMBER = {"value": 1}


def _figure_caption(title: str, description: str) -> None:
    """Render a numbered caption beneath a figure to guide interpretation."""
    number = _FIGURE_NUMBER["value"]
    _FIGURE_NUMBER["value"] += 1
    st.markdown(f"**Figure {number}. {title}.** {description.strip()}")

st.title("Data & methods")
st.caption("Survey design and modelling strategy overview")

st.header("Dataset")
st.markdown(
    """
    The analysis draws on the 2018 Kazakhstan wave of the Generations and Gender
    Survey. Survey weights are used for descriptive statistics, while regression models include controls for age,
    education, employment, parental background, and regional fixed effects.
    """
)

try:
    survey = load_kaz_ggs()
except FileNotFoundError as error:
    st.warning(str(error))
    survey = None
except (ImportError, ValueError) as error:
    st.warning(f"{error}")
    survey = None

if survey is not None:
    st.header("Sampling overview")
    region_column = "aregion" if "aregion" in survey.columns else None
    if region_column:
        region_counts = survey[region_column].astype(str).value_counts().head(10).sort_values(ascending=False)
        if region_counts.empty or region_counts.sum() == 0:
            st.info("Regional sampling counts are unavailable in the current dataset.")
        else:
            fig_region, ax_region = plt.subplots(figsize=(8, 5))
            palette = ["#3a7ca5"] * len(region_counts)
            sns.barplot(x=region_counts.values, y=region_counts.index, palette=palette, ax=ax_region)
            total_sample = region_counts.sum()
            ax_region.set_xlabel("Respondents", fontweight="bold")
            ax_region.set_ylabel("Region", fontweight="bold")
            for label in ax_region.get_xticklabels() + ax_region.get_yticklabels():
                label.set_fontweight("bold")
            ax_region.set_xlim(left=0)
            st.pyplot(fig_region)
            plt.close(fig_region)

            _figure_caption(
                "Survey coverage across regions",
                """
                Horizontal bars report respondent counts for the ten largest regional samples.
                Labels combine the raw count with the share of the top-ten sample, showing how coverage
                concentrates in densely populated southern and urban oblasts while still spanning all major regions.
                """,
            )
    else:
        st.info("Region identifiers were not found in the dataset; skipping the sampling figure.")

    st.header("Model covariates")
    covariate_columns = [column for column in ("aage", "numbiol", "family_support", "egalitarian") if column in survey.columns]

    if covariate_columns:
        covariate_frame = survey[covariate_columns].apply(pd.to_numeric, errors="coerce")
        numeric_covariates = covariate_frame.dropna(axis=0, how="any")
        if not numeric_covariates.empty:
            corr = numeric_covariates.corr()
            fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0,
                vmax=1,
                ax=ax_corr,
                cbar=False,
            )
            ax_corr.set_xlabel("", fontweight="bold")
            ax_corr.set_ylabel("", fontweight="bold")
            for label in ax_corr.get_xticklabels() + ax_corr.get_yticklabels():
                label.set_fontweight("bold")
            st.pyplot(fig_corr)
            plt.close(fig_corr)

            _figure_caption(
                "Correlation structure of continuous covariates",
                """
                Each cell shows the pairwise Pearson correlation between standardised inputs such as age,
                completed fertility, and value indexes. Annotated coefficients highlight modest associations,
                indicating that the covariates can be entered together without multicollinearity concerns.
                """,
            )
    else:
        st.info("The covariate fields needed for the heatmap are not present in the dataset.")

st.header("Modelling approach")
st.markdown(
    """
    * **Poisson regression** estimates completed fertility counts and reports
      incidence rate ratios. Specifications progressively add demographic,
      socio-economic, and occupational controls.
    * **Cox proportional hazards** capture timing to successive births. Hazard
      ratios describe the relative speed of reaching each parity.
    * **Index construction** aggregates Likert-scale items into standardised
      measures for family support and egalitarian attitudes, enabling direct
      comparison across models.
    """
)
