import joblib
import os

class ModelUtils:
    @staticmethod
    def save_model(model, filename: str):
        """
        Save the given model to disk using joblib.
        
        :param model: The model object to save.
        :param filename: The filename or path where the model will be saved.
        """
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save model
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename: str):
        """
        Load a model from disk using joblib.
        
        :param filename: The filename or path where the model is saved.
        :return: The loaded model or None if the file doesn't exist.
        """
        if not os.path.exists(filename):
            print(f"Error: Model file {filename} not found!")
            return None

        # Load model
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
