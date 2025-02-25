from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# Load trained model and dataset
rf_model = joblib.load("random_forest_model.pkl")  # Ensure this file exists
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
df = pd.read_csv("disease_symptom_data.csv")  # Ensure this file has correct columns

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms_input = data.get('symptoms', '').lower()  # Convert input to lowercase
    
    # Ensure dataset symptoms are also in lowercase for matching
    df['Symptoms'] = df['Symptoms'].str.lower()

    # Find matching disease based on symptoms
    matched_row = df[df['Symptoms'].apply(lambda x: set(symptoms_input.split(", ")).issubset(set(x.split(", "))))]

    if matched_row.empty:
        return jsonify({"error": "No matching disease found for given symptoms"}), 404
    
    # Extract details
    response = {
        "predicted_disease": matched_row.iloc[0]["Possible Disease"],
        "causes": matched_row.iloc[0]["Causes"],
        "precautions": matched_row.iloc[0]["Precautions"],
        "medicines": matched_row.iloc[0]["Medicines"]
    }
    
    return jsonify(response)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
