import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(page_title="About", layout="wide")

NUMERIC_FEATURES_DEFAULT = [
    "hour_studied",
    "pass_grades",
    "attendance_rate",
    "sleep_hour",
]

st.title("Student Grade Predictor")
st.write(
    "An app that predicts students grade from study features using the scikick sklearn models. Use the sidebar to navigate."
)
st.write(
    "The dataset used is from the UCI Machine Learning Repository. The dataset contains 395 instances and 33 attributes. The attributes are: gender, race/ethnicity, parental level of education, lunch, test preparation course"
)

st.write("## Technical Details")
st.write(
    "\nPages:\n"
    "• **Predict** – simple model (PolynomialFeatures + Ridge) using three features:                 studytime, failures, absences\n"
    "• **Dataset** – quick preview of the UCI Student Performance (Math) dataset\n\n"
    "Target: **G3** (final grade, 0–20)."
)

st.info("Tip: Use the sidebar to navigate between pages.")
