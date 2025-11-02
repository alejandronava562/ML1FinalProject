import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from utils import load_uci_model

st.set_page_config(page_title="Student Grade Predict Page", layout="wide")
st.title("Student Math Grade AI Predictor")

# --- Load the trained model --- #
df = load_uci_model()

FEATURES = ["studytime", "failures", "absences", "G2"]
TARGET = "G3"

missing = []
for col in FEATURES + [TARGET]:
    if col not in df.columns:
        missing.append(col)

if len(missing) > 0:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Keep only the relevant columns
data = df[FEATURES + [TARGET]].dropna().copy()
x = data[FEATURES].astype(float)
y = data[TARGET].astype(float)

def train_model(x, y):
    model = make_pipeline(PolynomialFeatures(2), Ridge())
    model.fit(x, y)
    return model

# call the model
model = train_model(x, y)

# --- Form for user input --- #
with st.form("predict_form"):
    st.write("### Please enter the following information:")
    st.write("---")
    
    st.write("Weekly study time: (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)")
    study_value = st.slider("Study slider", 1, 4)
    
    st.write("Number of past class failures (numeric: n if 1<=n<3, else     4)")
    failure_value = st.slider("Failure slider", 0, 4)
    
    st.write("Number of school absences (numeric: from 0 to 93)")
    absences_value = st.number_input("Absences:",min_value=0, max_value=93, value=4, step=1)
    
    st.write("Current grade on a 0-20 scale")
    current_value = st.number_input("Current Grade:",min_value=0, max_value=20, value=10, step=1)

    submitted = st.form_submit_button("Submit")
    
    # call the trained model to predict the grade
    if submitted == True:
        new_row = []
        new_row.append(float(study_value))
        new_row.append(float(failure_value))
        new_row.append(float(absences_value))
        new_row.append(float(current_value))

        # wrap into dataframe
        row_data = []
        row_data.append(new_row)
        row = pd.DataFrame(row_data, columns=FEATURES)

        
        #predict
        prediction = model.predict(row)
        pred_value = prediction[0]
        
        #clip values
        if pred_value < 0 :
            pred_value = 0
        if pred_value > 20:
            pred_value = 20

        st.success(f"Predicted Final Grade: **{round(pred_value, 2)} / 20**")
        if pred_value >= 14:
            status = "PASS"
        else:
            status = "AT RISK"

        st.write(f"Status: **{status}**")

        with st.expander("Your inputs"):
            st.write("Current Grade (G2):", current_value)
            st.write("Study Time:", study_value)
            st.write("Failures:", failure_value)
            st.write("Absences:", absences_value)
            


    