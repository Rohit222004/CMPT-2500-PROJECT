import mlflow.sklearn
import pandas as pd
import numpy as np

class ModelPredictor:
    def __init__(self, run_id: str):
        """
        Initialize the predictor with the MLflow run ID.
        
        :param run_id: MLflow run ID to load the logged model.
        """
        # Set MLflow tracking URI (ensure MLflow server is running)
        mlflow.set_tracking_uri("http://localhost:8080")  

        model_uri = f"runs:/{run_id}/model"
        self.model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from MLflow run ID: {run_id}")

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        :param input_data: DataFrame of input features.
        :return: numpy array of predictions.
        """
        return self.model.predict(input_data)

    def run_prediction(self, input_file: str, output_file: str):
        """
        Load input data from a CSV, predict, and save the combined output.
        
        :param input_file: Path to CSV containing features (e.g., X_test.csv).
        :param output_file: Path to save the CSV with predictions.
        """
        X_new = pd.read_csv(input_file)
        predictions = self.predict(X_new)
        predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
        combined_df = pd.concat([X_new.reset_index(drop=True), predictions_df], axis=1)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data with predictions saved to {output_file}")

if __name__ == "__main__":
    RUN_ID = "5632798a8687486c80f2ab50fbc1997c"  
    INPUT_FILE = "Test_Cases/X_test.csv"
    OUTPUT_FILE = "Predictions/model_predictions_combined.csv"

    predictor = ModelPredictor(RUN_ID)
    predictor.run_prediction(INPUT_FILE, OUTPUT_FILE)
