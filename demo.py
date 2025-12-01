import pandas as pd
import joblib
import os

# =========================================
# ⚙️ CONFIGURATION
# =========================================
MODEL_PATH = r"model/Best_model.pkl"

# =========================================
# 🛠️ HELPER FUNCTIONS
# =========================================
def load_model():
    """Loads the serialized machine learning pipeline."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("   Did you run the training notebook and save the model?")
        exit()
    
    print(f"⏳ Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!\n")
    return model

def make_prediction(model, applicant_data):
    """
    Accepts a dictionary of applicant data, converts it to a DataFrame,
    and returns the prediction (Approved/Rejected).
    """
    # 1. Convert dictionary to DataFrame (One single row)
    df = pd.DataFrame([applicant_data])
    
    # 2. Predict using the pipeline 
    # (The pipeline handles scaling, imputation, and encoding automatically)
    try:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] # Probability of Approval
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return

    # 3. Interpret Result
    status = "✅ APPROVED" if prediction == 1 else "❌ REJECTED"
    confidence = round(probability * 100, 2)
    
    # 4. Display Output
    print("-" * 40)
    print(f"👤 Applicant: {applicant_data['Gender']} | {applicant_data['Education']} | Credit: {applicant_data['Credit_History']}")
    print(f"💰 Income: ${applicant_data['Total_Income']} | Loan: ${applicant_data['Loan_Amount']}k")
    print(f"🔮 Decision: {status} (Confidence: {confidence}%)")
    print("-" * 40)

# =========================================
# 🚀 MAIN EXECUTION
# =========================================
if __name__ == "__main__":
    
    # 1. Load the Brain 🧠
    clf = load_model()

    # 2. Define Test Cases (Simulating User Input)
    
    # Case A: The "Ideal" Candidate (High Credit, Graduate, Moderate Income)
    customer_1 = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'Total_Income': 6000,       # Applicant + Coapplicant
        'Loan_Amount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,      # Good Credit
        'Property_Area': 'Semiurban'
    }

    # Case B: The "Risky" Candidate (Bad Credit, High Income)
    customer_2 = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '2',
        'Education': 'Graduate',
        'Self_Employed': 'Yes',
        'Total_Income': 8000,       # High Income
        'Loan_Amount': 200,
        'Loan_Amount_Term': 360,
        'Credit_History': 0.0,      # Bad Credit (The Gatekeeper!)
        'Property_Area': 'Urban'
    }

    # Case C: The "Borderline" Candidate (Low Income, Good Credit)
    customer_3 = {
        'Gender': 'Female',
        'Married': 'No',
        'Dependents': '0',
        'Education': 'Not Graduate',
        'Self_Employed': 'No',
        'Total_Income': 2500,       # Low Income
        'Loan_Amount': 100,
        'Loan_Amount_Term': 180,
        'Credit_History': 1.0,
        'Property_Area': 'Rural'
    }

    # 3. Run Predictions
    print("🤖 RUNNING PREDICTION SYSTEM...\n")
    
    make_prediction(clf, customer_1)
    make_prediction(clf, customer_2)
    make_prediction(clf, customer_3)