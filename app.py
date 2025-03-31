from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import random

app = Flask(__name__)
CORS(app)  # Allow React frontend to connect to Flask

# Load the trained model
try:
    model = joblib.load("fraud_model.pkl")  # Ensure model exists
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        if len(features) != 29:
            return jsonify({"error": f"Expected 29 features, received {len(features)}"}), 400

        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New route to return fraud data for the dashboard
@app.route('/fraud-data', methods=['GET'])
def get_fraud_data():
    sample_data = [
        {"date": "2025-03-29", "fraudulent": random.randint(10, 50), "legitimate": random.randint(500, 1000)},
        {"date": "2025-03-30", "fraudulent": random.randint(15, 55), "legitimate": random.randint(550, 1100)},
        {"date": "2025-03-31", "fraudulent": random.randint(20, 60), "legitimate": random.randint(600, 1200)},
    ]
    return jsonify(sample_data)

if __name__ == '__main__':
    app.run(debug=True)
