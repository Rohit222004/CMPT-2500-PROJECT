# evaluate.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, model_path: str, X_test_file: str, y_test_file: str):
        """
        Initialize with paths to the saved model and test datasets.
        
        :param model_path: Path to the saved model (e.g., 'ridge_model.pkl').
        :param X_test_file: Path to CSV file containing test features.
        :param y_test_file: Path to CSV file containing test target values.
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        # Load test features as a DataFrame (with headers)
        self.X_test = pd.read_csv(X_test_file)
        # Load test targets; squeeze() converts a single-column DataFrame to a Series
        self.y_test = pd.read_csv(y_test_file).squeeze()

    def evaluate(self):
        """
        Evaluate the model on the test set and print evaluation metrics.
        
        :return: Tuple containing (MSE, RMSE, R2 score)
        """
        # Pass the DataFrame directly to preserve column names for transformation.
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        print("Optimized Ridge Regression Model Evaluation on Test Data:")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R-squared: {r2}")
        return mse, rmse, r2

if __name__ == "__main__":
    evaluator = ModelEvaluator('models/ridge_model.pkl', 'Test_Cases/X_test.csv', 'Test_Cases/y_test.csv')
    evaluator.evaluate()
