import streamlit as st
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

# fetch dataset 


# # data (as pandas dataframes) 
# if student_performance and student_performance.data:
#     X = student_performance.data.features 
#     y = student_performance.data.targets
# else:
#     st.error("Failed to load dataset")
#     st.stop() 

# # metadata 
# print(student_performance.metadata) 

# # variable information 
# print(student_performance.variables) 

def load_csv(file_to_path):
    df = pd.read_csv(file_to_path, sep=";")
    df.columsn = [c.strip for c in df.columns]
    return df

st.set_page_config(page_title = "Student Grade Predictor",layout="wide")

NUMERIC_FEATURES_DEFAULT = [
  "hour_studied",
  "pass_grades",
  "attendance_rate",
  "sleep_hour"
]

st.title("Student Grade Predictor")
st.write("An app that predicts students grade from study features using the scikick sklearn models. Use the sidebar to navigate.")
st.write("The dataset used is from the UCI Machine Learning Repository. The dataset contains 395 instances and 33 attributes. The attributes are: gender, race/ethnicity, parental level of education, lunch, test preparation course")

st.write("## Technical Details")
st.write(
    "\nPages:\n"
    "• **Predict** – simple model (PolynomialFeatures + Ridge) using three features:                 studytime, failures, absences\n"
    "• **Dataset** – quick preview of the UCI Student Performance (Math) dataset\n\n"
    "Target: **G3** (final grade, 0–20).")

st.info(
    "Tip: Use the sidebar to navigate between pages."
)
