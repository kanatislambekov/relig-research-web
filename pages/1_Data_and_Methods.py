"""Data and methodology overview with supporting visuals."""
from __future__ import annotations

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from data_loader import load_kaz_ggs

sns.set_theme(style="whitegrid")

st.title("Data & methods")
st.caption("Survey design and modelling strategy overview")

st.header("Dataset")
st.markdown(
    """
    The analysis draws on the Kazakhstan wave of the Generations and Gender
    Survey. It covers adults aged 18–59 with detailed histories of partnership,
    fertility, and socio-economic background. Survey weights are used for
    descriptive statistics, while regression models include controls for age,
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
    st.header("Sampling footprint")
    region_column = "aregion" if "aregion" in survey.columns else None
    if region_column:
        region_counts = survey[region_column].astype(str).value_counts().head(10).sort_values(ascending=True)
        fig_region, ax_region = plt.subplots(figsize=(8, 5))
        sns.barplot(x=region_counts.values, y=region_counts.index, palette="crest", ax=ax_region)
        ax_region.set_xlabel("Respondents")
        ax_region.set_ylabel("Region")
        ax_region.set_xlim(left=0)
        ax_region.set_title("Top regions by sample size")
        st.pyplot(fig_region)
        plt.close(fig_region)

        st.markdown(
            """
            The survey spans all major oblasts, with larger representation from
            densely populated southern and urban regions. Balanced coverage
            ensures that regression estimates are not driven by a single region.
            """
        )

    st.header("Model covariates")
    covariate_columns = [column for column in ("aage", "numbiol", "family_support", "egalitarian") if column in survey.columns]

    if covariate_columns:
        covariate_frame = survey[covariate_columns].apply(pd.to_numeric, errors="coerce")
        numeric_covariates = covariate_frame.dropna(axis=0, how="any")
        if not numeric_covariates.empty:
            corr = numeric_covariates.corr()
            fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="flare", vmin=0, vmax=1, ax=ax_corr)
            ax_corr.set_title("Correlation among continuous covariates")
            st.pyplot(fig_corr)
            plt.close(fig_corr)

            st.markdown(
                """
                Covariates exhibit modest correlations, indicating that fertility
                models benefit from including both value indexes and demographic
                characteristics without severe multicollinearity.
                """
            )

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
