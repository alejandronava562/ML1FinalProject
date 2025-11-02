import streamlit as st
from utils import load_uci_model
import altair as alt
import pandas as pd

st.set_page_config(page_title="Dataset", layout="wide")
st.title("Dataset")

# load the dataset
df = load_uci_model()

# Preview + Shape
st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)
st.write("---")

st.write("### Shape")
st.write("Rows",df.shape[0])
st.write("Columns",df.shape[1])

# --- Scatterplot --- #

st.write("---")
df_copy = df.copy()
st.subheader("Final Grade VS Features")
x_options = []
options = ["absences","age","studytime","sex"]
for op in options:
  if op in df_copy.columns:
    x_options.append(op)

if len(x_options) > 0:
  x_feature = st.selectbox("Select an X-feature", x_options, index=0)
  work_df = df_copy.copy()
  if x_feature == "sex":
    if work_df["sex"].dtype == object:
      work_df["sex_num"] = work_df["sex"].map({"M": 0, "F": 1})
      x_feature_to_plot = "sex_num"
    else:
      x_feature_to_plot = "sex_num"
  else:
    x_feature_to_plot = x_feature

  tooltips = []
  base = ["G3", "studytime", "failures", "absences","age","sex"]
  for c in base :
    if c in work_df.columns:
      tooltips.append(c)
  plot = (alt.Chart(work_df).mark_circle().encode(
    x=alt.X(f"{x_feature_to_plot}:Q", title=x_feature),
    y=alt.Y("G3:Q",title = "Final Grade(0-20)"),
    tooltip=tooltips,
  ).interactive().properties(height=420)
  )
  pass_line_df = pd.DataFrame({"y": [10]})
  pass_line = alt.Chart(pass_line_df).mark_rule().encode(y="y")

st.altair_chart(plot)
      
    


## --- Correlation with G3 (Final Grade) --- ##

  