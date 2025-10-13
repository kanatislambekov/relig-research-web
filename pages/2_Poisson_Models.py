import altair as alt
import pandas as pd
import streamlit as st

from data_loader import count_stars, load_model_table, parse_numeric

st.title("Poisson regression models")

st.markdown(
    "These models estimate the expected number of children by modelling fertility counts with a Poisson link. "
    "Each table below focuses on a different set of covariates so that you can examine how the incidence rate ratio (IRR) "
    "changes as controls are added."
)


def _friendly_label(column: str) -> str:
    if column.startswith("("):
        parts = column.split(" ", 1)
        if len(parts) == 2:
            return parts[1]
    return column


def _render_table(file_name: str, description: str, key_prefix: str) -> pd.DataFrame:
    subtitle, table = load_model_table(file_name)
    st.subheader(subtitle or file_name)
    st.caption(description)

    if table.empty:
        st.warning("No data available for this specification.")
        return pd.DataFrame()

    value_columns = [col for col in table.columns if col not in {"Variable", "Context"}]
    default_selection = value_columns[: min(6, len(value_columns))]

    selected_columns = st.multiselect(
        "Columns to display",
        options=value_columns,
        default=default_selection,
        key=f"{key_prefix}-columns",
    )

    if not selected_columns:
        st.info("Select at least one model column to inspect.")
        return table

    preview = table[["Variable", "Context"] + selected_columns].copy()
    preview = preview.rename(columns={col: _friendly_label(col) for col in selected_columns})
    st.dataframe(preview, use_container_width=True)

    csv_bytes = preview.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download selection as CSV",
        data=csv_bytes,
        file_name=f"{key_prefix}_poisson.csv",
        mime="text/csv",
        key=f"{key_prefix}-download",
    )
    return table


def _plot_variable(table: pd.DataFrame, variable: str, title: str):
    if table.empty:
        return

    row = table[table["Variable"].str.lower() == variable.lower()]
    if row.empty:
        return

    row = row.iloc[0]
    data_rows = []
    for column in table.columns:
        if column in {"Variable", "Context"}:
            continue
        numeric_value = parse_numeric(row.get(column, ""))
        if numeric_value is None:
            continue
        data_rows.append(
            {
                "Model": _friendly_label(column),
                "IRR": numeric_value,
                "Significance": "*" * count_stars(row.get(column, "")),
            }
        )

    if not data_rows:
        return

    chart_df = pd.DataFrame(data_rows)
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Model", sort=None),
            y=alt.Y("IRR", title="Incidence rate ratio"),
            color=alt.value("#1f77b4"),
            tooltip=["Model", alt.Tooltip("IRR", format=".3f"), "Significance"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


with st.expander("Muslim and non-Muslim fertility comparison", expanded=True):
    muslim_table = _render_table(
        "Poisson_Muslim.xlsx",
        "Models with demographic, regional, and socioeconomic controls show the fertility gap by denomination.",
        key_prefix="poisson-muslim",
    )
    st.markdown("### Muslim coefficient across specifications")
    _plot_variable(muslim_table, "muslim", "Muslim coefficient")

with st.expander("Family and egalitarian values", expanded=False):
    values_table = _render_table(
        "Poisson_Values.xlsx",
        "These specifications add value-based indexes to understand how parental support and egalitarian attitudes correlate with fertility.",
        key_prefix="poisson-values",
    )
    st.markdown("### Egalitarian value coefficient")
    _plot_variable(values_table, "egalitarian", "Egalitarian coefficient")

with st.expander("Religiosity intensity", expanded=False):
    religiosity_table = _render_table(
        "Poisson_relig.xlsx",
        "Comparing categorical and continuous measures of religiosity clarifies whether more devout individuals have higher fertility.",
        key_prefix="poisson-relig",
    )
    st.markdown("### Effect of religiosity")
    _plot_variable(religiosity_table, "religiuos cat", "Religiosity coefficient")
