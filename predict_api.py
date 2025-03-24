from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd
from logging_config import configure_logging
import traceback

# Initialize logging
loggers = configure_logging()
logger = loggers['api']

app = Flask(__name__)

# Define paths for the models and preprocessed data
model_v1_path = os.path.join("models", "ridge_model_v1.pkl")
model_v2_path = os.path.join("models", "linear_model_v2.pkl")
preprocessed_data_path = os.path.join("Data", "preprocessed", "preprocessed_data.csv")

# Load models at startup using joblib.load()
model_v1, model_v2, preprocessed_data = None, None, None

try:
    model_v1 = joblib.load(model_v1_path)
    logger.info(f"Model v1 loaded successfully from {model_v1_path}, type: {type(model_v1)}")
    if hasattr(model_v1, 'named_steps'):
        logger.info(f"Model v1 steps: {model_v1.named_steps}")
except Exception as e:
    logger.error(f" Error loading model v1: {e}")

try:
    model_v2 = joblib.load(model_v2_path)
    logger.info(f" Model v2 loaded successfully from {model_v2_path}, type: {type(model_v2)}")
    if hasattr(model_v2, 'named_steps'):
        logger.info(f"Model v2 steps: {model_v2.named_steps}")
except Exception as e:
    logger.error(f" Error loading model v2: {e}")

# Load preprocessed data at startup
try:
    preprocessed_data = pd.read_csv(preprocessed_data_path)
    logger.info(f" Preprocessed data loaded successfully from {preprocessed_data_path}")
except FileNotFoundError:
    logger.warning(f" Preprocessed data file not found at {preprocessed_data_path}")
    preprocessed_data = None
except Exception as e:
    logger.error(f" Error loading preprocessed data: {e}")
    preprocessed_data = None

# Define the expected columns (order and names should match your training data)
EXPECTED_COLUMNS = [
    "model_year", "make", "model", "mileage", "price", "transmission_from_vin",
    "fuel_type_from_vin", "days_on_market", "msrp", "number_price_changes",
    "dealer_name", "listing_type", "listing_first_date", "vehicle_age", "year", "month", "day"
]

# ========================
# ROOT ENDPOINT
# ========================
@app.route('/', methods=['GET'])
def root():
    info = {
        "message": "Welcome to the CMPT-2500_PROJECT Prediction API",
        "description": "This API serves predictions for our ML models.",
        "endpoints": {
            "/CMPT-2500_PROJECT_home": "API home",
            "/v1/predict1": "Predict using Ridge model",
            "/v2/predict1": "Predict using Linear Regression model",
            "/CMPT-2500_PROJECT_health_status": "API health check"
        }
    }
    return jsonify(info)

# ========================
# API HOME ENDPOINT
# ========================
@app.route('/CMPT-2500_PROJECT_home', methods=['GET'])
def home():
    info = {
        "message": "Welcome to the CMPT-2500_PROJECT Prediction API.",
        "description": "Use /v1/predict1 for model v1 and /v2/predict1 for model v2.",
        "endpoints": {
            "/v1/predict1": "Predict using Ridge model",
            "/v2/predict1": "Predict using Linear Regression model",
            "/CMPT-2500_PROJECT_health_status": "API health check"
        }
    }
    return jsonify(info)

# ========================
# 🚀 HEALTH CHECK ENDPOINT
# ========================
@app.route('/CMPT-2500_PROJECT_health_status', methods=['GET'])
def health_status():
    status = {
        "status": "CMPT-2500_PROJECT API is up and running!",
        "models_loaded": {
            "model_v1": model_v1 is not None,
            "model_v2": model_v2 is not None
        }
    }
    logger.info("Health check successful")
    return jsonify(status)

# ========================
# 🚀 PREDICTION ENDPOINTS
# ========================
@app.route('/v1/predict1', methods=['GET'])
def predict_v1():
    if model_v1 is None:
        logger.error("❌ Model v1 is not available")
        return jsonify({"error": "Model v1 is not available"}), 500

    try:
        if preprocessed_data is None or preprocessed_data.empty:
            logger.warning("⚠️ Preprocessed data is not available or empty")
            return jsonify({"error": "Preprocessed data is not available"}), 500

        logger.info(f"📥 Using preprocessed data for prediction (v1): {preprocessed_data.shape}")

        # ✅ Make predictions for all rows
        predictions = model_v1.predict(preprocessed_data)
        logger.info(f"✅ Predictions (v1) successful for {len(predictions)} rows")

        # ✅ Output only the predictions — no input data
        response = predictions.tolist()

        logger.info(f"✅ Returning {len(response)} predictions")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Prediction error (v1): {traceback.format_exc()}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/v2/predict1', methods=['GET'])
def predict_v2():
    if model_v2 is None:
        logger.error("❌ Model v2 is not available")
        return jsonify({"error": "Model v2 is not available"}), 500

    try:
        if preprocessed_data is None or preprocessed_data.empty:
            logger.warning("⚠️ Preprocessed data is not available or empty")
            return jsonify({"error": "Preprocessed data is not available"}), 500

        logger.info(f"📥 Using preprocessed data for prediction (v2): {preprocessed_data.shape}")

        # ✅ Make predictions for all rows
        predictions = model_v2.predict(preprocessed_data)
        logger.info(f"✅ Predictions (v2) successful for {len(predictions)} rows")

        # ✅ Output only the predictions — no input data
        response = predictions.tolist()

        logger.info(f"✅ Returning {len(response)} predictions")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Prediction error (v2): {traceback.format_exc()}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# ========================
# 🚀 MAIN ENTRY POINT
# ========================
if __name__ == "__main__":
    logger.info("Starting Flask API...")
    app.run(host='0.0.0.0', port=9000, debug=True)
