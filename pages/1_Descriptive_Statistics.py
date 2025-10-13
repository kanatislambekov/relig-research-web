import altair as alt
import pandas as pd
import streamlit as st

from data_loader import count_stars, load_model_table, parse_numeric

st.title("Descriptive statistics and value indexes")

subtitle, indexes = load_model_table("Indexes.xlsx")
if subtitle:
    st.subheader(subtitle)

if indexes.empty:
    st.warning("The indexes table could not be loaded. Confirm that the Excel file is available in the Results folder.")
    st.stop()

st.markdown(
    "The table below summarises how denominational membership relates to egalitarian and family-support value indexes. "
    "Use the column selector to focus on the indexes that are most relevant for your discussion."
)

value_columns = [col for col in indexes.columns if col not in {"Variable", "Context"}]

def _friendly_label(column: str) -> str:
    if column.startswith("("):
        parts = column.split(" ", 1)
        if len(parts) == 2:
            return parts[1]
    return column

suggested_defaults = [col for col in value_columns if any(token in col.lower() for token in ("egalitarian", "parents", "children", "family"))]
if not suggested_defaults:
    suggested_defaults = value_columns

selected_columns = st.multiselect(
    "Columns to display",
    options=value_columns,
    default=suggested_defaults,
)

if not selected_columns:
    st.info("Select at least one column to visualise the regression outputs.")
    st.stop()

preview = indexes[["Variable", "Context"] + selected_columns].copy()
preview = preview.rename(columns={col: _friendly_label(col) for col in selected_columns})

st.dataframe(preview, use_container_width=True)

csv_bytes = preview.to_csv(index=False).encode("utf-8")
st.download_button("Download selected columns as CSV", data=csv_bytes, file_name="indexes_preview.csv", mime="text/csv")

muslim_row = indexes[indexes["Variable"].str.lower() == "muslim1"]
if not muslim_row.empty:
    muslim_row = muslim_row.iloc[0]
    chart_rows = []
    for column in value_columns:
        numeric_value = parse_numeric(muslim_row.get(column, ""))
        if numeric_value is None:
            continue
        chart_rows.append(
            {
                "Index": _friendly_label(column),
                "Coefficient": numeric_value,
                "Significance": "*" * count_stars(muslim_row.get(column, "")),
            }
        )

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows)
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Coefficient", title="Coefficient"),
                y=alt.Y("Index", sort="-x"),
                color=alt.condition(alt.datum.Coefficient > 0, alt.value("#1f77b4"), alt.value("#d62728")),
                tooltip=["Index", alt.Tooltip("Coefficient", format=".3f"), "Significance"],
            )
        )
        st.markdown("### Muslim coefficients across value indexes")
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Positive coefficients indicate stronger agreement with the corresponding value statement among Muslim respondents."
        )

st.info(
    "Significance stars follow conventional thresholds: *p* < 0.1 (*), *p* < 0.05 (**), *p* < 0.01 (***)."
)
