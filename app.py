import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load("model/Best_model.pkl")

st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details to check loan eligibility.")

# --- INPUT SECTION ---
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
total_income = st.number_input("Total Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 360])
credit_history = st.selectbox("Credit History (1 = Good | 0 = Bad)", [0, 1])
property_area = st.selectbox("Property Area", ["Rural", "Urban", "Semiurban"])

# --- Create DataFrame for Prediction ---
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'Total_Income': [total_income],
    'Loan_Amount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# --- PREDICT BUTTON ---
if st.button("Predict 🚀"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
