"""Narrative results combining regression outputs with visual summaries."""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from data_loader import load_model_table, parse_numeric

sns.set_theme(style="whitegrid")

st.title("Results")
st.caption("Regression evidence on fertility, values, and religiosity")


def _safe_load_table(file_name: str) -> Tuple[str, pd.DataFrame]:
    try:
        return load_model_table(file_name)
    except FileNotFoundError as error:
        st.error(str(error))
        return "", pd.DataFrame()


def _extract_coefficients(row: pd.Series, rename_map: Dict[str, str]) -> pd.DataFrame:
    records = []
    for column, label in rename_map.items():
        value = parse_numeric(row.get(column, ""))
        if value is None:
            continue
        records.append({"Model": label, "Effect": value})
    return pd.DataFrame(records)


def _styled_data_table(title: str, table: pd.DataFrame) -> None:
    if title:
        st.subheader(title)
    else:
        st.subheader("Regression table")
    if table.empty:
        st.warning("The regression table is empty after tidying.")
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)


# --- Poisson models: denomination effects -----------------------------------
poisson_muslim_title, poisson_muslim = _safe_load_table("Poisson_Muslim.xlsx")
if poisson_muslim.empty:
    st.stop()

_styled_data_table(poisson_muslim_title or "Poisson regression: denomination effects", poisson_muslim)

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
            respondents even after extensive socio-demographic controls. The
            incidence rate ratio stays above one across specifications, signalling
            higher expected counts of biological children and supporting the
            denomination-based interpretation described in the study memo.
            """
        )


# --- Poisson models: value orientations --------------------------------------
poisson_values_title, poisson_values = _safe_load_table("Poisson_Values.xlsx")
if not poisson_values.empty:
    _styled_data_table(poisson_values_title or "Poisson regression: value orientations", poisson_values)

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
            egalitarian attitudes drive the incidence rate ratio below one. As
            highlighted in the memo, once denominational controls are added the
            positive association of family support diminishes, while egalitarian
            attitudes remain a consistent drag on completed fertility.
            """
        )


# --- Poisson models: religiosity intensity -----------------------------------
poisson_relig_title, poisson_relig = _safe_load_table("Poisson_relig.xlsx")
if not poisson_relig.empty:
    _styled_data_table(poisson_relig_title or "Poisson regression: religiosity levels", poisson_relig)

    base_levels = poisson_relig[
        poisson_relig["Variable"].str.contains("RECODE of a1112", case=False)
        & ~poisson_relig["Variable"].str.contains("Std. Err.", case=False)
    ]
    interaction_rows = poisson_relig[
        poisson_relig["Variable"].str.contains("1.muslim1#", case=False)
        & ~poisson_relig["Variable"].str.contains("Std. Err.", case=False)
    ]

    records = []
    for category in range(1, 6):
        label = f"Level {category}"
        base_effect = 1.0
        if category > 1:
            matches = base_levels[base_levels["Variable"].str.contains(f"= {category}")]
            if not matches.empty:
                base_value = parse_numeric(matches.iloc[0]["(1) Religiuos cat"])
                if base_value is not None:
                    base_effect = base_value
        muslim_multiplier = 1.0
        if category > 1:
            interactions = interaction_rows[interaction_rows["Variable"].str.contains(f"#{category}.")]
            if not interactions.empty:
                extra = parse_numeric(interactions.iloc[0]["(1) Religiuos cat"])
                if extra is not None:
                    muslim_multiplier = extra
        records.append({"Religiosity": label, "Group": "Non-Muslim", "IRR": base_effect})
        records.append({"Religiosity": label, "Group": "Muslim", "IRR": base_effect * muslim_multiplier})

    religiosity_effects = pd.DataFrame(records)
    religiosity_effects["Religiosity"] = pd.Categorical(
        religiosity_effects["Religiosity"], categories=[f"Level {i}" for i in range(1, 6)], ordered=True
    )

    fig_relig, ax_relig = plt.subplots(figsize=(9, 4))
    sns.lineplot(
        data=religiosity_effects,
        x="Religiosity",
        y="IRR",
        hue="Group",
        style="Group",
        markers=True,
        dashes=False,
        ax=ax_relig,
    )
    ax_relig.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax_relig.set_ylabel("Incidence rate ratio")
    ax_relig.set_title("Fertility response to religiosity intensity")
    st.pyplot(fig_relig)
    plt.close(fig_relig)

    st.markdown(
        """
        Categorical religiosity boosts fertility for both denominations, with the
        steepest rise concentrated among Muslims at the upper end of the scale.
        The interaction terms underline that strongly religious Muslims retain a
        markedly higher expected number of children even after controls.
        """
    )


# --- Cox hazard models: denomination differences -----------------------------
cox_muslim_title, cox_muslim = _safe_load_table("Cox_muslim.xlsx")
if not cox_muslim.empty:
    _styled_data_table(cox_muslim_title or "Cox hazard model: denomination effects", cox_muslim)

    cox_rows = cox_muslim[cox_muslim["Variable"].str.strip().str.lower() == "muslim1"]
    if cox_rows.empty:
        st.warning("Could not find the Muslim hazard ratios in the Cox models table.")
    else:
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
                Hazard ratios from the Cox models reveal that Muslim households
                reach each successive birth order more quickly, with the largest
                gap appearing at the transition to the third child.
                """
            )


# --- Cox hazard models: religiosity effects ----------------------------------
cox_rel_title, cox_rel = _safe_load_table("Cox_rel.xlsx")
if not cox_rel.empty:
    _styled_data_table(cox_rel_title or "Cox hazard model: religiosity", cox_rel)

    base_levels = cox_rel[
        cox_rel["Variable"].str.contains(".religious", case=False)
        & ~cox_rel["Variable"].str.contains("#", regex=False)
        & ~cox_rel["Variable"].str.contains("Std. Err.", case=False)
    ]
    interaction_levels = cox_rel[
        cox_rel["Variable"].str.contains("1.muslim1#", case=False)
        & ~cox_rel["Variable"].str.contains("Std. Err.", case=False)
    ]

    birth_columns = {
        "1_birth .": "First birth",
        "2_birth .": "Second birth",
        "3_birth .": "Third birth",
    }

    hazard_records = []
    for category in range(1, 6):
        label = f"Level {category}"
        base_row = (
            base_levels[base_levels["Variable"].str.startswith(f"{category}.")]
            if category > 1
            else pd.DataFrame()
        )
        interaction_row = (
            interaction_levels[interaction_levels["Variable"].str.contains(f"#{category}.")]
            if category > 1
            else pd.DataFrame()
        )

        for column, birth_label in birth_columns.items():
            base_effect = 1.0
            if category > 1 and not base_row.empty:
                numeric = parse_numeric(base_row.iloc[0][column])
                if numeric is not None:
                    base_effect = numeric
            muslim_effect = base_effect
            if category > 1 and not interaction_row.empty:
                numeric = parse_numeric(interaction_row.iloc[0][column])
                if numeric is not None:
                    muslim_effect *= numeric
            hazard_records.append(
                {"Religiosity": label, "Group": "Non-Muslim", "Birth order": birth_label, "Hazard ratio": base_effect}
            )
            hazard_records.append(
                {"Religiosity": label, "Group": "Muslim", "Birth order": birth_label, "Hazard ratio": muslim_effect}
            )

    hazard_frame = pd.DataFrame(hazard_records)
    hazard_frame["Religiosity"] = pd.Categorical(
        hazard_frame["Religiosity"], categories=[f"Level {i}" for i in range(1, 6)], ordered=True
    )

    g = sns.relplot(
        data=hazard_frame,
        x="Religiosity",
        y="Hazard ratio",
        hue="Group",
        style="Group",
        kind="line",
        markers=True,
        dashes=False,
        col="Birth order",
        facet_kws={"sharey": True},
        height=3.6,
        aspect=1.1,
    )
    for ax in g.axes.flatten():
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_ylim(bottom=0)
    g.fig.suptitle("Birth timing response to religiosity", y=1.02)
    st.pyplot(g.fig)
    plt.close(g.fig)

    st.markdown(
        """
        Religious commitment accelerates first births among non-Muslims and has a
        markedly stronger effect for Muslim households at higher birth orders. The
        interaction terms echo the memo's interpretation that sustained religious
        engagement amplifies parity progression for Muslim families while the
        effect is muted elsewhere.
        """
    )
