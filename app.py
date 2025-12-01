import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
# Ensure the path is correct based on your folder structure
model = joblib.load("model/Best_model.pkl")

st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details to check loan eligibility.")

# --- INPUT SECTION ---
# Layout using columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    total_income = st.number_input("Total Income ($)", min_value=0, value=5000)
    loan_amount = st.number_input("Loan Amount ($) in Thousands (eg: 56k)", min_value=0, value=120)
    loan_term = st.selectbox("Loan Amount Term (Months)", [480, 360, 300, 240, 180, 120, 84, 60, 36, 12])
    
    credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)")
    
    property_area = st.selectbox("Property Area", ["Rural", "Urban", "Semiurban"])

# --- PREDICT BUTTON ---
if st.button("Predict 🚀"):
    # Create DataFrame
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'Total_Income': [total_income],
        'Loan_Amount': [loan_amount],
        # Ensure object type matches training for Term
        'Loan_Amount_Term': [loan_term], 
        # CRITICAL FIX: Ensure Credit History is float (1.0 not 1)
        'Credit_History': [float(credit_history)], 
        'Property_Area': [property_area]
    })

    # DEBUGGING: Uncomment the line below to see what data is being sent to the model
    # st.write(input_data)

    try:
        # Get Probability (Confidence)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # Probability of Approval

        if total_income >= 100:
            st.error(f"❌ Loan Rejected. Too low income!!!")
        elif prediction == 1:
            st.success(f"🎉 Loan Approved! (Confidence: {probability*100:.2f}%)")
            st.balloons()
        else:
            st.error(f"❌ Loan Rejected. (Risk Score: {(1-probability)*100:.2f}%)")
            
            # Explain why (Simple Logic Check)
            if credit_history == 0.0:
                st.warning("⚠️ Note: Poor Credit History is a major factor in rejection.")
                
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
