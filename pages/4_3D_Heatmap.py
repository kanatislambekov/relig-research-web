"""3D heatmap summarising effect sizes across key regression variables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  Needed for 3D plotting

from data_loader import load_model_table, parse_numeric

sns.set_theme(style="darkgrid")

st.title("3D heatmap")
st.caption("Comparing effect sizes across models and variables")

# Load required tables
_, poisson_muslim = load_model_table("Poisson_Muslim.xlsx")
_, poisson_values = load_model_table("Poisson_Values.xlsx")

# Helper to extract coefficients

def _grab_coefficients(table: pd.DataFrame, variable: str, columns: dict[str, str]) -> pd.Series:
    row = table[table["Variable"].str.strip().str.lower() == variable.lower()]
    if row.empty:
        return pd.Series(dtype=float)
    row = row.iloc[0]
    values = {}
    for column, label in columns.items():
        value = parse_numeric(row.get(column, ""))
        if value is not None:
            values[label] = value
    return pd.Series(values)


model_labels = ["Baseline", "Socio-demo", "Full"]

# Muslim IRRs mapped to baseline -> age controls, socio-demo -> full + region, full -> full control
muslim_effect = _grab_coefficients(
    poisson_muslim,
    "muslim",
    {
        "(1) IRR_age": "Baseline",
        "(3) IRR_full control - reg": "Socio-demo",
        "(4) IRR_full control": "Full",
    },
)

family_effect = _grab_coefficients(
    poisson_values,
    "family_support",
    {
        "(10) Both": "Baseline",
        "(11) Both": "Socio-demo",
        "(12) Both": "Full",
    },
)

egalitarian_effect = _grab_coefficients(
    poisson_values,
    "egalitarian",
    {
        "(7) Egalitarian": "Baseline",
        "(8) Egalitarian": "Socio-demo",
        "(9) Egalitarian": "Full",
    },
)

# Assemble matrix
matrix = pd.DataFrame(
    {
        "Variable": ["Muslim identity", "Family support index", "Egalitarian index"],
    }
)
matrix = matrix.set_index("Variable")
matrix.loc["Muslim identity", model_labels] = muslim_effect.reindex(model_labels)
matrix.loc["Family support index", model_labels] = family_effect.reindex(model_labels)
matrix.loc["Egalitarian index", model_labels] = egalitarian_effect.reindex(model_labels)
matrix = matrix.astype(float)

if matrix.isna().all().all():
    st.warning("Insufficient data to construct the 3D heatmap.")
    st.stop()

matrix = matrix.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")

x_labels = matrix.index.tolist()
y_labels = model_labels
x_pos, y_pos = np.meshgrid(np.arange(len(x_labels)), np.arange(len(y_labels)), indexing="ij")

x_pos = x_pos.flatten()
y_pos = y_pos.flatten()
z_pos = np.zeros_like(x_pos, dtype=float)
dx = dy = 0.5
dz = matrix.values.flatten()

norm = plt.Normalize(dz.min(), dz.max())
cmap = plt.get_cmap("viridis")
colors = cmap(norm(dz))

bars = ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # Required for colorbar


ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)
ax.set_xticks(np.arange(len(x_labels)) + dx / 2)
ax.set_xticklabels(x_labels, rotation=20, ha="right")
ax.set_yticks(np.arange(len(y_labels)) + dy / 2)
ax.set_yticklabels(y_labels)
ax.set_zlabel("Effect size (ratio)")
ax.set_zlim(bottom=0)
ax.set_title("Effect magnitudes across models")

mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array(dz)


st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
    The heatmap highlights the contrast between the positive fertility effects
    of Muslim identification and family-support values versus the suppressing
    influence of egalitarian attitudes. Intensity fades but never reverses after
    adding full controls, signalling robust relationships across specifications.
    """
)
