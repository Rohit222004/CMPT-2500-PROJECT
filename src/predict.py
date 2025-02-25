import joblib
import pandas as pd
import numpy as np

class ModelPredictor:
    def __init__(self, model_path: str):
        """
        Initialize the predictor with the path to the saved model.
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

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
    MODEL_PATH = 'models/ridge_model.pkl'
    INPUT_FILE = 'Test_Cases/X_test.csv'
    OUTPUT_FILE = 'Predictions/model_predictions_combined.csv'

    predictor = ModelPredictor(MODEL_PATH)
    predictor.run_prediction(INPUT_FILE, OUTPUT_FILE)
