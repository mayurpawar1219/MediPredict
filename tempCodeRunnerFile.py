from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained model and encoders
rf_model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset for additional details
df = pd.read_csv("disease_symptom_data.csv")  # Ensure it has columns: Disease, Causes, Precautions, Medicines

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms_input = data.get("symptoms", "")

        input_vector = vectorizer.transform([symptoms_input])
        predicted_label = rf_model.predict(input_vector)
        predicted_disease = label_encoder.inverse_transform(predicted_label)[0]

        # Retrieve additional info from the dataset
        disease_info = df[df["Disease"] == predicted_disease].iloc[0]  # Get first matching row

        response = {
            "predicted_disease": predicted_disease,
            "causes": disease_info["Causes"],
            "precautions": disease_info["Precautions"],
            "medicines": disease_info["Medicines"]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
