import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, run_id: str, X_test_file: str, y_test_file: str):
        """
        Initialize with MLflow run ID and test dataset paths.

        :param run_id: MLflow run ID to load the logged model.
        :param X_test_file: Path to CSV file containing test features.
        :param y_test_file: Path to CSV file containing test target values.
        """
        model_uri = f"runs:/{run_id}/model"
        self.model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from MLflow run ID: {run_id}")

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

        print("Optimized Ridge Regression Model Evaluation on Test Data:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")

        return mse, rmse, r2

if __name__ == "__main__":
    # Set MLflow tracking URI (adjust if running on a different server)
    mlflow.set_tracking_uri("http://localhost:8080")  
    RUN_ID = "5632798a8687486c80f2ab50fbc1997c"
    X_TEST_FILE = "Test_Cases/X_test.csv"
    Y_TEST_FILE = "Test_Cases/y_test.csv"

    evaluator = ModelEvaluator(RUN_ID, X_TEST_FILE, Y_TEST_FILE)
    evaluator.evaluate()
