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
sex = st.selectbox("Select the value for Sex", ["0 for male", "1 for female"])
if sex == "0 for male":
    sex = 0
else:
    sex = 1

age = st.number_input("Enter the value for Age: ")
bmi = st.number_input("Enter the value for BMI: ")
hunt_and_hess = st.number_input("Enter the value for Hunt and Hess Scale score: ")
modified_fisher = st.number_input("Enter the value for Modified Fisher Scale score: ")

hypertension = st.selectbox("Enter the value for Hypertension", ["0 for No", "1 for Yes"])
if hypertension == "0 for No":
    hypertension = 0
else:
    hypertension = 1

smoking = st.selectbox("Enter the value for Smoking", ["0 for No", "1 for Yes"])
if smoking == "0 for No":
    smoking = 0
else:
    smoking = 1

size_of_aneurysm = st.number_input("Enter the value for Size of Aneurysm in mm: ")
treatment_modality = st.selectbox("Enter the value for Treatment Modality", ["1 for Microsurgical Clipping", "2 for Endovascular Coiling"])
if treatment_modality == "1 for Microsurgical Clipping":
    treatment_modality = 1
else:
    treatment_modality = 2

mrs_at_discharge = st.number_input("Enter the value for mRS score at discharge: ")

cerebral_infarction = st.selectbox("Enter the value for Cerebral Infarction", ["0 for No", "1 for Yes"])
if cerebral_infarction == "0 for No":
    cerebral_infarction = 0
else:
    cerebral_infarction = 1

monocyte_value = st.number_input("Enter the value for Monocyte laboratory value: ")

if st.button("Predict"):
    # Create a new data point using the user-entered values
    new_data = [[sex, age, bmi, hunt_and_hess, modified_fisher, hypertension, smoking, 
                 size_of_aneurysm, treatment_modality, mrs_at_discharge, cerebral_infarction, 
                 monocyte_value]]
    
    # Make a prediction for the new data point
    prediction = clf.predict(new_data)
    prediction_proba = clf.predict_proba(new_data)
    
    # Get the prediction probability
    probability = prediction_proba[0][prediction[0]]
    
    # Print the prediction probability as a percentage
    st.write("Probability of delayed cerebral ischemia: ", probability*100, "%")

