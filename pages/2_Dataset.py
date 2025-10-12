import streamlit as st

from utils import load_uci_model

df = load_uci_model()
st.write(df.head())
