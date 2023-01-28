import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Delayed Cerebral Ischemia Prediction", page_icon=":guardsman:", layout="wide")

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/drazzam/machine_learning_tools/main/delayed_cerebral_ischemia/data.csv")

# Impute missing values and remove rows with invalid values
df.dropna(inplace=True)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# Split the data into features and target
X = df[["sex", "age", "bmi", "hh_score", "mfisher", "htn", "smoke", "size_mm", "tx", "mrs_discharge", "infarct", "monocytes"]]
y = df["dci"]

# Train the model
model = xgb.XGBRegressor(random_state=0)
model.fit(X, y)

st.title("Delayed Cerebral Ischemia Prediction App")

# Get user input for the 12 variables
sex = st.selectbox("Select gender", ["Male", "Female"])
sex = 0 if sex == "Male" else 1
age = st.number_input("Enter age", min_value=0, max_value=120, value=50)
bmi = st.number_input("Enter BMI", min_value=0, max_value=50, value=30)
hh_score = st.number_input("Enter Hunt and Hess score", min_value=0, max_value=5, value=3)
mfisher = st.number_input("Enter modified Fisher scale", min_value=0, max_value=5, value=3)
htn = st.selectbox("Hypertension", ["Yes", "No"])
htn = 1 if htn == "Yes" else 0
smoke = st.selectbox("Smoking", ["Yes", "No"])
smoke = 1 if smoke == "Yes" else 0
size_mm = st.number_input("Enter size of aneurysm in mm", min_value=0, max_value=100, value=25)
tx = st.selectbox("Treatment modality", ["Microsurgical Clipping", "Endovascular Coiling"])
tx = 1 if tx == "Microsurgical Clipping" else 2
mrs_discharge = st.number_input("Enter mrs at discharge", min_value=0, max_value=5, value=2)
infarct = st.selectbox("Cerebral infarction", ["Yes", "No"])
infarct = 1 if infarct == "Yes" else 0
monocytes = st.number_input("Enter monocyte count in 10^3/uL: ")

# Create a dataframe from the user input
user_input = pd.DataFrame(
{'sex': sex,
'age': age,
'bmi': bmi,
'hh_score': hh_score,
'mfisher': mfisher,
'htn': htn,
'smoke': smoke,
'size_mm': size_mm,
'tx': tx,
'mrs_discharge': mrs_discharge,
'infarct': infarct
}, index=[0])

# Use the model to make a prediction
prediction = model.predict(user_input)

#Show the prediction
st.write("The Probability of Developing Delayed Cerebral Ischemia Is: %", prediction*100[0])

# Add a button to rerun the app
if st.button("Predict again"):
 pass

# Run the Streamlit app
if name == "main":
 st.run()


