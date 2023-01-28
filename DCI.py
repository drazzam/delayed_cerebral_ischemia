import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/drazzam/machine_learning_tools/main/delayed_cerebral_ischemia/data.csv")

# Impute missing values and remove rows with invalid values
data.dropna(inplace=True)
data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

# Define the features and target variable
X = data[["sex", "age", "bmi", "hh_score", "mfisher", "htn", "smoke", "size_mm", "tx", "mrs_discharge", "infarct", "monocytes" ]]
y = data["dci"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Let the user enter values for the variables
sex = st.selectbox("Select The Gender:", ["Male", "Female"])
if sex == "Male":
    sex = 0
else:
    sex = 1

age = st.number_input("Enter The Age: ")
bmi = st.number_input("Enter The Body Mass Index: ")
hunt_and_hess = st.number_input("Enter The Hunt and Hess Scale score: ")
modified_fisher = st.number_input("Enter The Modified Fisher Scale score: ")

hypertension = st.selectbox("Is The Patient Hypertensive?", ["No", "Yes"])
if hypertension == "No":
    hypertension = 0
else:
    hypertension = 1

smoking = st.selectbox("Does The Patient Smoke?", ["No", "Yes"])
if smoking == "No":
    smoking = 0
else:
    smoking = 1

size_of_aneurysm = st.number_input("Enter The Size of the Aneurysm in mm: ")
treatment_modality = st.selectbox("Enter the value for Treatment Modality", ["Microsurgical Clipping", "Endovascular Coiling"])
if treatment_modality == "Microsurgical Clipping":
    treatment_modality = 1
else:
    treatment_modality = 2

mrs_at_discharge = st.number_input("Enter The mRS score at Discharge: ")

cerebral_infarction = st.selectbox("Did The Patient Get Cerebral Infarction?", ["No", "Yes"])
if cerebral_infarction == "No":
    cerebral_infarction = 0
else:
    cerebral_infarction = 1

monocyte_value = st.number_input("Enter The Value For Monocyte (10^3/uL): ")

if st.button("Predict"):
    # Create a new data point using the user-entered values
    new_data = [[sex, age, bmi, hunt_and_hess, modified_fisher, hypertension, smoking, 
                 size_of_aneurysm, treatment_modality, mrs_at_discharge, cerebral_infarction, 
                 monocyte_value]]
    
    # Make a prediction for the new data point
    prediction = clf.predict(new_data)
    prediction_proba = clf.predict_proba(new_data)
    
    # Get the prediction probability
    prediction = int(prediction)
    probability = prediction_proba[0,prediction].item()

    # Print the prediction probability as a percentage
    st.write("The Probability of Developing Delayed Cerebral Ischemia: ", probability*100, "%")

