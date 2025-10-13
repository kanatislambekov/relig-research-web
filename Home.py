import streamlit as st
from data_loader import count_stars, load_docx_paragraphs, load_model_table, parse_numeric

st.set_page_config(
    page_title="Kazakhstan Fertility & Religiosity Explorer",
    page_icon="📈",
    layout="wide",
)

st.title("Kazakhstan Fertility & Religiosity Explorer")

intro_paragraphs = load_docx_paragraphs("Text (1).docx")
if intro_paragraphs:
    st.markdown(
        "\n\n".join(intro_paragraphs[:2])
        if len(intro_paragraphs) > 1
        else intro_paragraphs[0]
    )
else:
    st.info("Upload the extended abstract to populate this introduction.")

st.divider()

# Load highlighted metrics from the regression tables.
_, poisson_muslim = load_model_table("Poisson_Muslim.xlsx")
_, poisson_values = load_model_table("Poisson_Values.xlsx")
_, indexes_table = load_model_table("Indexes.xlsx")

col1, col2, col3 = st.columns(3)

with col1:
    if not poisson_muslim.empty:
        muslim_row = poisson_muslim[poisson_muslim["Variable"].str.lower() == "muslim"]
        if not muslim_row.empty:
            muslim_row = muslim_row.iloc[0]
            irr_column = next(
                (col for col in muslim_row.index if "full control" in col.lower() and "other" not in col.lower()),
                None,
            )
            irr_value = parse_numeric(muslim_row.get(irr_column, "")) if irr_column else None
            stars = count_stars(muslim_row.get(irr_column, "")) if irr_column else 0
            if irr_value:
                lift = (irr_value - 1.0) * 100
                st.metric(
                    label="Muslim vs. non-Muslim fertility",
                    value=f"{irr_value:.3f} IRR",
                    delta=f"{lift:+.1f}% expected children",
                    help=(
                        "The incidence rate ratio from the full-control model shows that Muslim respondents "
                        "have a higher expected number of children."
                        + (" Significance: " + "*" * stars if stars else "")
                    ),
                )
            else:
                st.metric("Muslim vs. non-Muslim fertility", "Data unavailable")
        else:
            st.metric("Muslim vs. non-Muslim fertility", "Data unavailable")
    else:
        st.metric("Muslim vs. non-Muslim fertility", "Missing table")

with col2:
    if not poisson_values.empty:
        egalitarian_row = poisson_values[poisson_values["Variable"].str.lower() == "egalitarian"]
        if not egalitarian_row.empty:
            egalitarian_row = egalitarian_row.iloc[0]
            egalitarian_column = next(
                (col for col in egalitarian_row.index if "egalitarian" in col.lower()),
                None,
            )
            irr_value = parse_numeric(egalitarian_row.get(egalitarian_column, "")) if egalitarian_column else None
            stars = count_stars(egalitarian_row.get(egalitarian_column, "")) if egalitarian_column else 0
            if irr_value:
                change = (irr_value - 1.0) * 100
                st.metric(
                    label="Impact of egalitarian values",
                    value=f"{irr_value:.3f} IRR",
                    delta=f"{change:+.1f}% fertility shift",
                    help=(
                        "Coefficients less than 1 indicate that more egalitarian attitudes correlate with "
                        "lower fertility." + (" Significance: " + "*" * stars if stars else "")
                    ),
                )
            else:
                st.metric("Impact of egalitarian values", "Data unavailable")
        else:
            st.metric("Impact of egalitarian values", "Data unavailable")
    else:
        st.metric("Impact of egalitarian values", "Missing table")

with col3:
    if not indexes_table.empty:
        muslim_index = indexes_table[indexes_table["Variable"].str.lower() == "muslim1"]
        if not muslim_index.empty:
            muslim_index = muslim_index.iloc[0]
            family_column = next(
                (col for col in muslim_index.index if "family" in col.lower()),
                None,
            )
            coeff = parse_numeric(muslim_index.get(family_column, "")) if family_column else None
            stars = count_stars(muslim_index.get(family_column, "")) if family_column else 0
            if coeff is not None:
                st.metric(
                    label="Muslim advantage on family support index",
                    value=f"{coeff:.3f}",
                    delta="Higher support among Muslim respondents",
                    help=(
                        "Positive coefficients show stronger support for intergenerational obligations."
                        + (" Significance: " + "*" * stars if stars else "")
                    ),
                )
            else:
                st.metric("Muslim advantage on family support index", "Data unavailable")
        else:
            st.metric("Muslim advantage on family support index", "Data unavailable")
    else:
        st.metric("Muslim advantage on family support index", "Missing table")

st.divider()

st.header("Navigate the research")

st.markdown(
    """
- **Descriptive statistics** summarises how value indexes vary across denominations.
- **Poisson regression models** explore expected number of children under different controls.
- **Cox proportional hazards** look at the timing of births across religious and value groups.
- **Extended abstract** reproduces the narrative accompanying the quantitative evidence.
    """
)

st.success("Use the sidebar to switch between pages and explore the analysis in detail.")
