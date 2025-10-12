import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Student Grade Predict Page", layout="wide")
st.title("Student Grade AI Predictor")

# --- Load the trained model --- #
@st.cache_data
def load_uci_model():
    ds = fetch_ucirepo(id=320)
    x = ds.data.features
    y = ds.data.targets
    df = pd.concat([x,y], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

df = load_uci_model()
st.write(df.head())


# --- Form for user input --- #
with st.form("predict_form"):
    st.write("Please enter the following information:")
    st.write(
        "Weekly study time: (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours,            or 4 - >10 hours)"
    )
    slider_val = st.slider("Study slider", 1, 4)
    st.write("Number of past class failures (numeric: n if 1<=n<3, else     4)")
    slider_val = st.slider("Failure slider", 1, 4)
    st.write("Number of school absences (numeric: from 0 to 93)")
    st.text_input("Number of absences:")

    submitted = st.form_submit_button("Submit")
    # call the trained model to predict the grade




