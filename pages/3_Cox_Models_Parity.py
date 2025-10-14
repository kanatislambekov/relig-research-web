import altair as alt
import pandas as pd
import streamlit as st

from data_loader import count_stars, load_model_table, parse_numeric

st.title("Cox proportional hazard models")

st.markdown(
    "The Cox models reveal how quickly different groups progress to each birth order. "
    "Hazard ratios greater than one imply a higher likelihood of experiencing the event sooner, while values below one denote slower timing."
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
    default_selection = value_columns[: min(5, len(value_columns))]

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
        file_name=f"{key_prefix}_cox.csv",
        mime="text/csv",
        key=f"{key_prefix}-download",
    )
    return table


def _plot_variable(table: pd.DataFrame, variable: str):
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
                "Hazard ratio": numeric_value,
                "Significance": "*" * count_stars(row.get(column, "")),
            }
        )

    if not data_rows:
        return

    chart_df = pd.DataFrame(data_rows)
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Model", sort=None),
            y=alt.Y("Hazard ratio", title="Hazard ratio"),
            color=alt.condition(alt.datum["Hazard ratio"] > 1, alt.value("#2ca02c"), alt.value("#d62728")),
            tooltip=["Model", alt.Tooltip("Hazard ratio", format=".2f"), "Significance"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


with st.expander("Birth timing by denomination", expanded=True):
    muslim_table = _render_table(
        "Cox_muslim.xlsx",
        "Baseline and fully controlled models highlighting differences between Muslim and non-Muslim respondents.",
        key_prefix="cox-muslim",
    )
    st.markdown("### Muslim hazard ratios across birth orders")
    _plot_variable(muslim_table, "muslim1")

with st.expander("Religiosity and value interactions", expanded=False):
    religiosity_table = _render_table(
        "Cox_rel.xlsx",
        "These models incorporate categorical religiosity measures and their interaction with Muslim identification.",
        key_prefix="cox-relig",
    )
    st.markdown("### Effect of being Muslim within religiosity models")
    _plot_variable(religiosity_table, "1.muslim1")

st.caption("Values above 1 (green) indicate faster progression to the birth, whereas red bars point to slower timing relative to the reference group.")
