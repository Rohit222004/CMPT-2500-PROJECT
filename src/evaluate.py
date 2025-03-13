import os
import joblib
import argparse
import pandas as pd
import numpy as np
import mlflow.sklearn
import mlflow
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, model_source: str, X_test_file: str, y_test_file: str, model_version: str = "v1"):
        """
        Initialize with either an MLflow run ID or local file path and test dataset paths.

        :param model_source: Either a run_id (if loading from MLflow) or a local file path.
        :param X_test_file: Path to CSV file containing test features.
        :param y_test_file: Path to CSV file containing test target values.
        :param model_version: Model version ("v1" for Ridge, "v2" for Linear Regression)
        """
        self.model_version = model_version

        # If model_source is a file path (exists on disk), load locally; otherwise, treat it as a run ID.
        if os.path.exists(model_source):
            self.model = joblib.load(model_source)
            print(f"Model loaded from local file: {model_source}")
        else:
            mlflow.set_tracking_uri("http://localhost:8080")
            model_uri = f"runs:/{model_source}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            print(f"Model loaded from MLflow run ID: {model_source}")

        # Load test data
        self.X_test = pd.read_csv(X_test_file)
        self.y_test = pd.read_csv(y_test_file).values.ravel()  # Ensure y_test is a 1D array

    def evaluate(self):
        """
        Evaluate the model on the test set and print evaluation metrics.

        :return: Tuple containing (MSE, RMSE, R2 score)
        """
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, predictions)

        if self.model_version == "v1":
            print("Optimized Ridge Regression Model Evaluation on Test Data:")
        elif self.model_version == "v2":
            print("Optimized Linear Regression Model Evaluation on Test Data:")

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")

        return mse, rmse, r2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument(
        "--model_version", 
        type=str, 
        choices=["v1", "v2"], 
        default="v1", 
        help="Model version to evaluate (v1 for Ridge, v2 for Linear Regression)"
    )
    parser.add_argument(
        "--run_id", 
        type=str, 
        default=None,
        help=("MLflow run id to load the model from MLflow (if not using local file). "
              "If not provided, a default run id is used based on model_version: "
              "v1: bbd4ea7aabab4c88b6e1f5c9f7a7c700, v2: 72f5986fc51c45f2bf3cd601fdf21111")
    )
    parser.add_argument(
        "--X_test_file", 
        type=str, 
        default="Test_Cases/X_test.csv", 
        help="Path to CSV file containing test features"
    )
    parser.add_argument(
        "--y_test_file", 
        type=str, 
        default="Test_Cases/y_test.csv", 
        help="Path to CSV file containing test target values"
    )
    parser.add_argument(
        "--use_local", 
        action="store_true", 
        help="Flag to indicate loading model from a local pickle file instead of MLflow"
    )
    parser.add_argument(
        "--both_models",
        action="store_true",
        help="Evaluate metrics for both models (v1 and v2)"
    )
    args = parser.parse_args()

    if args.both_models:
        # Evaluate both models: v1 (Ridge) and v2 (Linear Regression)
        models_info = {
            "v1": {
                "name": "Ridge Regression",
                "run_id": "bbd4ea7aabab4c88b6e1f5c9f7a7c700",  # default run id for v1
                "local_path": "models/ridge_model_v1.pkl"
            },
            "v2": {
                "name": "Linear Regression",
                "run_id": "72f5986fc51c45f2bf3cd601fdf21111",    # default run id for v2
                "local_path": "models/linear_model_v2.pkl"
            }
        }

        if args.run_id:
            print("Warning: run_id argument is ignored when --both_models is used. Using default run ids for each model.")

        for version, info in models_info.items():
            if args.use_local:
                model_source = info["local_path"]
            else:
                model_source = info["run_id"]
            print(f"\nEvaluating {info['name']} (model version: {version})...")
            evaluator = ModelEvaluator(model_source, args.X_test_file, args.y_test_file, version)
            evaluator.evaluate()
            print("\n" + "="*50 + "\n")
    else:
        # Single model evaluation
        if not args.run_id and not args.use_local:
            default_run_ids = {
                "v1": "bbd4ea7aabab4c88b6e1f5c9f7a7c700",
                "v2": "72f5986fc51c45f2bf3cd601fdf21111"
            }
            args.run_id = default_run_ids[args.model_version]

        if args.use_local:
            if args.model_version == "v1":
                model_source = "models/ridge_model_v1.pkl"
            else:
                model_source = "models/linear_model_v2.pkl"
        else:
            model_source = args.run_id

        evaluator = ModelEvaluator(model_source, args.X_test_file, args.y_test_file, args.model_version)
        evaluator.evaluate()
