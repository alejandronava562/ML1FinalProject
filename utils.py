import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo


@st.cache_data
def load_uci_model():
    ds = fetch_ucirepo(id=320)
    x = ds.data.features
    y = ds.data.targets
    df = pd.concat([x, y], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df
