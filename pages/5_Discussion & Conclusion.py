"""Discussion and conclusion narrative for the research site."""
from __future__ import annotations

import streamlit as st

st.title("Discussion & Conclusion")

st.header("Findings")
st.markdown(
    """
    The convergence of descriptive and modelling evidence demonstrates that
    religious identity and value orientations jointly shape fertility outcomes
    in Kazakhstan. Muslim households report more biological children, progress
    more rapidly to higher birth orders, and maintain stronger family-support
    norms. Egalitarian attitudes, in contrast, lower expected fertility even when
    demographic and regional controls are introduced.
    """
)

st.header("Implications")
st.markdown(
    """
    * **Policy relevance:** Family-support norms correlate with higher fertility,
      suggesting that social policies reinforcing intergenerational support could
      sustain higher birth rates without undermining autonomy.
    * **Value heterogeneity:** The persistence of egalitarian effects highlights
      emerging value pluralism. Future programmes should acknowledge that gender
      equality initiatives may moderate fertility intentions.
    * **Demographic momentum:** Younger age profiles within the Muslim
      population imply ongoing demographic momentum that will influence labour
      markets, education needs, and housing demand.
    """
)

st.header("Future research")
st.markdown(
    """
    * Extend the analysis with longitudinal GGS waves to observe whether value
      shifts precede changes in completed fertility.
    * Investigate regional policy experiments to identify contextual factors
      that amplify or dampen the influence of religiosity on fertility timing.
    * Collect qualitative interviews to explore how couples reconcile egalitarian
      ideals with family-support expectations in everyday decision-making.
    """
)


