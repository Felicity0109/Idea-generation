from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re

# ----------------------------
# Topic table
# ----------------------------
if topics:
    topics_df = pd.DataFrame(topics)
    st.dataframe(topics_df, use_container_width=True, hide_index=True)

    # Assign dominant topic per idea
if W is not None:
    dom = np.argmax(W, axis=1)
    work["Topic"] = (dom + 1).astype(int)

    # Compute topic composition by group
    comp = work.groupby(["Topic", group_col]).size().unstack(fill_value=0)

    # st.bar_chart expects numeric DataFrame; topics as index, groups as columns
    st.subheader("Topic composition by group")
    st.bar_chart(comp)
else:
    st.info("Topic model could not be fitted (insufficient text or unique terms).")


st.markdown("---")

# ----------------------------
# Searchable table + download
# ----------------------------
st.subheader("Idea Catalogue")

q1, q2 = st.columns([2, 1])
with q1:
    search = st.text_input("Search (title/description)", "")
with q2:
    sel_group = st.multiselect(
        "Filter by group",
        sorted(work[group_col].fillna("Unknown").unique().tolist())
    )

filtered = work.copy()

if search:
    s = search.lower()
    mask = filtered["__text__"].str.contains(re.escape(s), case=False, na=False)
    filtered = filtered[mask]

if sel_group:
    filtered = filtered[filtered[group_col].isin(sel_group)]

show_cols = [title_col, desc_col, group_col, date_col]
if "Topic" in filtered.columns:
    show_cols.append("Topic")

st.dataframe(
    filtered[show_cols].sort_values(by=date_col, ascending=False),
    use_container_width=True
)

# Download buttons
csv = filtered[show_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered (CSV)",
    data=csv,
    file_name="ideas_filtered.csv",
    mime="text/csv"
)

# ----------------------------
# Footer / Help
# ----------------------------
with st.expander("Deployment Tips"):
    st.markdown(
        """
Developed by SASOL Research and Technology (2025) ©
"""
    )
