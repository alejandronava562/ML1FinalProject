import streamlit as st
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
student_performance = fetch_ucirepo(id=320) 

# data (as pandas dataframes) 
if student_performance and student_performance.data:
    X = student_performance.data.features 
    y = student_performance.data.targets
else:
    st.error("Failed to load dataset")
    st.stop() 

# # metadata 
# print(student_performance.metadata) 

# # variable information 
# print(student_performance.variables) 



st.set_page_config(page_title = "Student Grade Predictor",layout="wide")

NUMERIC_FEATURES_DEFAULT = [
  "hour_studied",
  "pass_grades",
  "attendance_rate",
  "sleep_hour"
]

TARGET_COL = "final_grade"
#test#
# --------------------------------------------------------------
# Side Bar Controls
# --------------------------------------------------------------
st.sidebar.title("Options")
st.sidebar.markdown("---")


  
st.title("Student Grade Predictor")
st.write("An app that predicts students grade from study features using the scikick sklearn models. Use the sidebar to navigate.")
    