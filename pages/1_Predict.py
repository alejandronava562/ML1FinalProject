import streamlit as st
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from utils import load_uci_model

st.set_page_config(page_title="Student Grade Predict Page", layout="wide")
st.title("Student Grade AI Predictor")

# --- Load the trained model --- #
df = load_uci_model()

FEATURES = ["studytime", "failures", "absences"]
TARGET = "G3"


def train_model(x, y):
    model = make_pipeline(PolynomialFeatures(2), Ridge())
    model.fit(x, y)
    return model


# --- Form for user input --- #
with st.form("predict_form"):
    st.write("Please enter the following information:")
    st.write(
        "Weekly study time: (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours,            or 4 - >10 hours)"
    )
    slider_val = st.slider("Study slider", 1, 4)
    st.write(
        "Number of past class failures (numeric: n if 1<=n<3, else     4)")
    slider_val = st.slider("Failure slider", 1, 4)
    st.write("Number of school absences (numeric: from 0 to 93)")
    st.text_input("Number of absences:")

    submitted = st.form_submit_button("Submit")
    # call the trained model to predict the grade
