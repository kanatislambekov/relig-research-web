"""Narrative results combining regression outputs with visual summaries."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from data_loader import load_model_table, parse_numeric

sns.set_theme(style="whitegrid")

st.title("Results")
st.caption("Regression evidence on fertility, values, and religiosity")


def _extract_coefficients(row: pd.Series, rename_map: Dict[str, str]) -> pd.DataFrame:
    records = []
    for column, label in rename_map.items():
        value = parse_numeric(row.get(column, ""))
        if value is None:
            continue
        records.append({"Model": label, "Effect": value})
    return pd.DataFrame(records)


# --- Poisson models: denomination effects -----------------------------------
try:
    _, poisson_muslim = load_model_table("Poisson_Muslim.xlsx")
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

muslim_row = poisson_muslim[poisson_muslim["Variable"].str.strip().str.lower() == "muslim"]
if muslim_row.empty:
    st.warning("Could not locate the Muslim coefficient in the Poisson models table.")
else:
    muslim_row = muslim_row.iloc[0]
    rename_map = {
        "(1) IRR_age": "Age controls",
        "(3) IRR_full control - reg": "Full + region",
        "(4) IRR_full control": "Full controls",
        "(5) IRR_full control+occupation": "Full + occupation",
    }
    muslim_effects = _extract_coefficients(muslim_row, rename_map)

    if not muslim_effects.empty:
        fig_muslim, ax_muslim = plt.subplots(figsize=(8, 4))
        sns.barplot(data=muslim_effects, x="Model", y="Effect", color="#4C78A8", ax=ax_muslim)
        ax_muslim.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax_muslim.set_ylim(bottom=0)
        ax_muslim.set_ylabel("Incidence rate ratio")
        ax_muslim.set_title("Muslim fertility advantage across model specifications")
        for index, (_, row) in enumerate(muslim_effects.iterrows()):
            ax_muslim.text(index, row["Effect"] + 0.02, f"{row['Effect']:.2f}", ha="center")
        st.pyplot(fig_muslim)
        plt.close(fig_muslim)

        st.markdown(
            """
            The Poisson models confirm a persistent fertility premium for Muslim
            respondents even after introducing extensive socio-economic controls.
            The incidence rate ratio stays above one across specifications,
            indicating higher expected counts of biological children.
            """
        )

# --- Value indexes -----------------------------------------------------------
_, poisson_values = load_model_table("Poisson_Values.xlsx")
value_rows = poisson_values.set_index("Variable")

value_map = {
    "parents_support": {
        "(1) Parental help": "Parental help",
        "(2) Parental help": "Parental help + socio-demo",
        "(3) Parental help": "Parental help + full",
    },
    "children_support": {
        "(4) Children help": "Children help",
        "(5) Children help": "Children help + socio-demo",
        "(6) Children help": "Children help + full",
    },
    "family_support": {
        "(10) Both": "Combined support",
        "(11) Both": "Combined + socio-demo",
        "(12) Both": "Combined + full",
    },
    "egalitarian": {
        "(7) Egalitarian": "Egalitarian",
        "(8) Egalitarian": "Egalitarian + socio-demo",
        "(9) Egalitarian": "Egalitarian + full",
    },
}

value_frames = []
for variable, rename_map in value_map.items():
    if variable not in value_rows.index:
        continue
    coeffs = _extract_coefficients(value_rows.loc[variable], rename_map)
    if coeffs.empty:
        continue
    coeffs["Variable"] = variable.replace("_", " ").title()
    value_frames.append(coeffs)

if value_frames:
    combined_values = pd.concat(value_frames, ignore_index=True)
    fig_values, ax_values = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=combined_values,
        x="Model",
        y="Effect",
        hue="Variable",
        ax=ax_values,
    )
    ax_values.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax_values.set_ylim(bottom=0)
    ax_values.set_ylabel("Incidence rate ratio")
    ax_values.set_title("Value indexes and expected number of children")
    ax_values.legend(title="Index")
    st.pyplot(fig_values)
    plt.close(fig_values)

    st.markdown(
        """
        Family-support norms push fertility expectations upward, whereas
        egalitarian attitudes drive the incidence rate ratio below one. The
        negative association for egalitarian views remains even after adding
        extensive controls.
        """
    )

# --- Timing models -----------------------------------------------------------
_, cox_table = load_model_table("Cox_muslim.xlsx")
cox_rows = cox_table[cox_table["Variable"].str.strip().str.lower() == "muslim1"]
if not cox_rows.empty:
    cox_row = cox_rows.iloc[0]
    rename_map = {
        "1_birth .": "1st birth",
        "2_birth .": "2nd birth",
        "3_birth .": "3rd birth",
        "4_birth .": "4th birth",
    }
    hazards = _extract_coefficients(cox_row, rename_map)
    if not hazards.empty:
        fig_hazard, ax_hazard = plt.subplots(figsize=(8, 4))
        sns.pointplot(data=hazards, x="Model", y="Effect", color="#F58518", ax=ax_hazard)
        ax_hazard.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax_hazard.set_ylim(bottom=0)
        ax_hazard.set_ylabel("Hazard ratio")
        ax_hazard.set_title("Progression to higher birth orders for Muslim households")
        for index, (_, row) in enumerate(hazards.iterrows()):
            ax_hazard.text(index, row["Effect"] + 0.02, f"{row['Effect']:.2f}", ha="center")
        st.pyplot(fig_hazard)
        plt.close(fig_hazard)

        st.markdown(
            """
            Hazard ratios from the Cox models reveal that Muslim households reach
            each successive birth order more quickly, with the largest gap at the
            transition to the third child.
            """
        )
else:
    st.warning("Could not find the Muslim hazard ratios in the Cox models table.")
