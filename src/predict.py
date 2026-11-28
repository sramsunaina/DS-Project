import joblib
import pandas as pd

# Load saved model
model = joblib.load("models/loan_model.pkl")

# Example input
sample_data = pd.DataFrame([{
    "income_annum": 4100000,
    "loan_amount": 12200000,
    "loan_term": 8,
    "cibil_score": 417,
    "residential_assets_value": 12200000,
    "commercial_assets_value": 2200000,
    "luxury_assets_value": 8800000,
    "bank_asset_value": 8800000
}]) 

# Predict
prediction = model.predict(sample_data)

print("Loan Approval Prediction:", prediction[0])