from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from waitress import serve

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend to connect

# Load model and scaler
print("📁 Loading model and scaler...")
model = tf.keras.models.load_model("loan_model.keras")
scaler = joblib.load("scaler.pkl")
print("✅ Model and scaler loaded successfully!")

# Feature list
feature_names = [
    "Loan Amount", "Funded Amount", "Funded Amount Investor", "Term",
    "Batch Enrolled", "Interest Rate", "Grade", "Sub Grade", "Employment Duration",
    "Home Ownership", "Verification Status", "Payment Plan", "Debit to Income",
    "Delinquency - two years", "Inquires - six months", "Open Account",
    "Public Record", "Revolving Balance", "Revolving Utilities", "Total Accounts",
    "Initial List Status", "Total Received Interest", "Total Received Late Fee",
    "Recoveries", "Collection Recovery Fee", "Collection 12 months Medical",
    "Application Type", "Last week Pay", "Accounts Delinquent",
    "Total Collection Amount", "Total Current Balance", "Total Revolving Credit Limit"
]

# Categorical columns to encode
categorical_cols = [
    "Batch Enrolled", "Grade", "Sub Grade", "Employment Duration",
    "Verification Status", "Payment Plan", "Initial List Status", "Application Type"
]

@app.route('/')
def home():
    return "Welcome to the Loan Prediction API! Send a POST request to /predict with 32 features."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print(f"Received features: {len(data['features'])} - {data['features']}")

        if len(data['features']) != 32:
            return jsonify({'error': 'Input must contain exactly 32 features'}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data['features']], columns=feature_names)

        # Encode categorical columns
        for col in categorical_cols:
            input_df[col] = input_df[col].astype("category").cat.codes

        # Scale features
        features_scaled = scaler.transform(input_df.values)

        # Predict
        prediction_prob = model.predict(features_scaled)[0][0]
        result = "Approved" if prediction_prob > 0.5 else "Rejected"

        return jsonify({
            'prediction': result,
            'probability': float(prediction_prob)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Entry point with Waitress
if __name__ == '__main__':
    print("🚀 Starting server with Waitress...")
    serve(app, host='0.0.0.0', port=5000, threads=4)
