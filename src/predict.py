import os
import joblib
import argparse
import pandas as pd
import numpy as np

import mlflow.sklearn
import mlflow

class ModelPredictor:
    def __init__(self, model_path: str = None, run_id: str = None):
        """
        Initialize the predictor by loading the model either from a local file or from an MLflow run.

        :param model_path: Path to the locally saved pickle model.
        :param run_id: MLflow run ID to load the logged model.
        """
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} does not exist.")
            self.model = joblib.load(model_path)
            print(f"Model loaded from local file: {model_path}")
        elif run_id:
            mlflow.set_tracking_uri("http://localhost:8080")
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            print(f"Model loaded from MLflow run ID: {run_id}")
        else:
            raise ValueError("Either model_path or run_id must be provided.")

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        :param input_data: DataFrame of input features.
        :return: numpy array of predictions.
        """
        return self.model.predict(input_data)

    def run_prediction(self, input_file: str, output_file: str):
        """
        Load input data from a CSV, run prediction, and save the results combined with the input features.
        
        :param input_file: Path to CSV containing features.
        :param output_file: Path to save the CSV with predictions.
        """
        X_new = pd.read_csv(input_file)
        predictions = self.predict(X_new)
        predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
        combined_df = pd.concat([X_new.reset_index(drop=True), predictions_df], axis=1)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data with predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Prediction Script")
    parser.add_argument(
        "--model_version", 
        type=str, 
        choices=["v1", "v2"],
        default="v1", 
        help="Model version to use for prediction (v1 for Ridge, v2 for Linear Regression)"
    )
    parser.add_argument(
        "--run_id", 
        type=str, 
        default=None,
        help=("MLflow run id to load the model from MLflow (if not using local file). " 
              "If not provided, a default run id is used based on model_version: "
              "v1: 5ad98a2ff5f144de96b43d896b019636, v2: 2131e7520e174a3995e0f57b3fbfda78")
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="Test_Cases/X_test.csv", 
        help="Path to input CSV file with features"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="Predictions/model_predictions_combined.csv", 
        help="Base path for output CSV file for combined predictions. A suffix will be added if running both models."
    )
    parser.add_argument(
        "--use_local", 
        action="store_true", 
        help="Flag indicating to load the model from a local pickle file instead of MLflow"
    )
    parser.add_argument(
        "--both_models", 
        action="store_true", 
        help="If set, run predictions for both the Ridge (v1) and Linear Regression (v2) models and output separate files."
    )
    args = parser.parse_args()

    if args.both_models:
        # Load the input data once
        X_new = pd.read_csv(args.input_file)
        
        # Define model settings for Ridge and Linear Regression
        models_info = {
            "ridge": {
                "version": "v1",
                "run_id": "5ad98a2ff5f144de96b43d896b019636",  # Ridge model run id
                "local_path": "models/ridge_model_v1.pkl",
                "output_suffix": "ridge"
            },
            "linear": {
                "version": "v2",
                "run_id": "2131e7520e174a3995e0f57b3fbfda78",   # Linear Regression model run id
                "local_path": "models/linear_model_v2.pkl",
                "output_suffix": "linear"
            }
        }
        
        if args.run_id:
            print("Warning: run_id argument is ignored when --both_models is used. Using default run ids for each model.")
        
        for model_key, info in models_info.items():
            if args.use_local:
                predictor = ModelPredictor(model_path=info["local_path"])
            else:
                predictor = ModelPredictor(run_id=info["run_id"])
            
            predictions = predictor.predict(X_new)
            predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
            combined_df = pd.concat([X_new.reset_index(drop=True), predictions_df], axis=1)
            
            # Append the model suffix to the base output file name
            base, ext = os.path.splitext(args.output_file)
            output_file = f"{base}_{info['output_suffix']}{ext}"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            combined_df.to_csv(output_file, index=False)
            print(f"Combined data with predictions for {model_key} saved to {output_file}")
    else:
        # Single model prediction mode
        if not args.run_id:
            default_run_ids = {
                "v1": "5ad98a2ff5f144de96b43d896b019636",  # Ridge model run id
                "v2": "2131e7520e174a3995e0f57b3fbfda78"   # Linear Regression model run id
            }
            args.run_id = default_run_ids[args.model_version]
        
        if args.use_local:
            if args.model_version == "v1":
                model_file = "models/ridge_model_v1.pkl"
            elif args.model_version == "v2":
                model_file = "models/linear_model_v2.pkl"
            predictor = ModelPredictor(model_path=model_file)
        else:
            predictor = ModelPredictor(run_id=args.run_id)
        
        predictor.run_prediction(args.input_file, args.output_file)