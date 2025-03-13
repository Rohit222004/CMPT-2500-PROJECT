from flask import Flask, jsonify
import os
import joblib
import pandas as pd

app = Flask(__name__)

# Define paths for the models and preprocessed data
model_v1_path = os.path.join("models", "ridge_model_v1.pkl")
model_v2_path = os.path.join("models", "linear_model_v2.pkl")
preprocessed_data_path = os.path.join("Data", "preprocessed", "preprocessed_data.csv")  # Path to preprocessed data

# Load models at startup using joblib.load()
try:
    model_v1 = joblib.load(model_v1_path)
    print(f"Model v1 loaded successfully from {model_v1_path}, type: {type(model_v1)}")
    if hasattr(model_v1, 'named_steps'):
        print(f"Model v1 steps: {model_v1.named_steps}")
except Exception as e:
    model_v1 = None
    print(f"Error loading model v1: {e}")

try:
    model_v2 = joblib.load(model_v2_path)
    print(f"Model v2 loaded successfully from {model_v2_path}, type: {type(model_v2)}")
    if hasattr(model_v2, 'named_steps'):
        print(f"Model v2 steps: {model_v2.named_steps}")
except Exception as e:
    model_v2 = None
    print(f"Error loading model v2: {e}")

# Load preprocessed data at startup
try:
    # Load the preprocessed_data.csv file
    preprocessed_data = pd.read_csv(preprocessed_data_path)
    print(f"Preprocessed data loaded successfully from {preprocessed_data_path}")
    print("First few rows of preprocessed data:")
    print(preprocessed_data.head())
except Exception as e:
    preprocessed_data = None
    print(f"Error loading preprocessed data: {e}")

# Define the expected columns (order and names should match your training data)
EXPECTED_COLUMNS = [
    "model_year", "make", "model", "mileage", "price", "transmission_from_vin",
    "fuel_type_from_vin", "days_on_market", "msrp", "number_price_changes",
    "dealer_name", "listing_type", "listing_first_date", "vehicle_age", "year", "month", "day"
]
EXPECTED_FEATURES = len(EXPECTED_COLUMNS)

@app.route('/CMPT-2500_PROJECT_home', methods=['GET'])
def home():
    """
    Home endpoint that describes the API usage.
    """
    info = {
        "message": "Welcome to the CMPT-2500_PROJECT Prediction API.",
        "description": (
            "This API serves predictions for our ML models. "
            "Use /v1/predict1 for model v1 and /v2/predict1 for model v2."
        ),
        "endpoints": {
            "/v1/predict1": {
                "description": "Predict endpoint using model v1 (e.g., Ridge Regression)",
            },
            "/v2/predict1": {
                "description": "Predict endpoint using model v2 (e.g., Linear Regression)",
            },
            "/CMPT-2500_PROJECT_health_status": {
                "description": "Health check endpoint to verify API status"
            }
        }
    }
    return jsonify(info)

@app.route('/CMPT-2500_PROJECT_health_status', methods=['GET'])
def health_status():
    """
    Health endpoint to check if the API is running.
    """
    return jsonify({"status": "CMPT-2500_PROJECT API is up and running!"})

@app.route('/v1/predict1', methods=['GET'])
def predict_v1():
    """
    Predict endpoint for model v1.
    Uses preprocessed data from the Data/preprocessed folder.
    """
    if preprocessed_data is None:
        return jsonify({"error": "Preprocessed data is not available"}), 500

    if model_v1 is None:
        return jsonify({"error": "Model v1 is not available"}), 500

    try:
        # Ensure the preprocessed data has the expected columns
        if list(preprocessed_data.columns) != EXPECTED_COLUMNS:
            return jsonify({"error": f"Preprocessed data must have columns: {EXPECTED_COLUMNS}"}), 500

        # Make predictions
        predictions = model_v1.predict(preprocessed_data)
        print("Predictions from model v1:")
        print(predictions)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    return jsonify({"predictions": predictions.tolist()}), 200

@app.route('/v2/predict1', methods=['GET'])
def predict_v2():
    """
    Predict endpoint for model v2.
    Uses preprocessed data from the Data/preprocessed folder.
    """
    if preprocessed_data is None:
        return jsonify({"error": "Preprocessed data is not available"}), 500

    if model_v2 is None:
        return jsonify({"error": "Model v2 is not available"}), 500

    try:
        # Ensure the preprocessed data has the expected columns
        if list(preprocessed_data.columns) != EXPECTED_COLUMNS:
            return jsonify({"error": f"Preprocessed data must have columns: {EXPECTED_COLUMNS}"}), 500

        # Make predictions
        predictions = model_v2.predict(preprocessed_data)
        print("Predictions from model v2:")
        print(predictions)
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    return jsonify({"predictions": predictions.tolist()}), 200

if __name__ == "__main__":
    print("Starting Flask API...")
    app.run(host='127.0.0.1', port=9992, debug=True)