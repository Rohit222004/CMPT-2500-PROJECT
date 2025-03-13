import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
import joblib

from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature


class ModelTrainer:
    def __init__(
        self,
        preprocessed_file: str,
        model_output_path_v1: str,  # path for first model pkl
        model_output_path_v2: str,  # path for second model pkl
        test_cases_path: str
    ):
        """
        Initialize with:
          - path to the preprocessed CSV file
          - two model save paths (for v1 and v2)
          - test cases folder path
        """
        self.preprocessed_file = preprocessed_file
        self.model_output_path_v1 = model_output_path_v1
        self.model_output_path_v2 = model_output_path_v2
        self.test_cases_path = test_cases_path

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.best_model_v1 = None  # Will store the trained v1 model (Ridge)
        self.best_model_v2 = None  # Will store the trained v2 model (Linear Regression)

    def load_data(self) -> pd.DataFrame:
        """
        Load the preprocessed CSV data.
        """
        df = pd.read_csv(self.preprocessed_file)
        return df

    def prepare_dataset(self):
        """
        Prepare the dataset by splitting into features and target, then train/test split.
        """
        df = self.load_data()
        # Adjust columns based on your preprocessed data structure
        X = df[
            [
                "number_price_changes",
                "vehicle_age",
                "mileage",
                "price",
                "msrp",
                "dealer_name",
                "listing_type",
                "make",
                "model",
                "year",
                "month",
                "day",
            ]
        ]
        y = df["days_on_market"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data split into training and test sets.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_pipeline(self, regressor_class):
        """
        Build a pipeline given a particular regressor class (e.g. Ridge or LinearRegression).
        """
        numerical_features = [
            "mileage",
            "price",
            "msrp",
            "vehicle_age",
            "year",
            "month",
            "day",
            "number_price_changes",
        ]
        categorical_features = ["dealer_name", "listing_type", "make", "model"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer()),  # strategy tuned below
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numerical_features,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        # Instantiate the pipeline with a chosen regressor
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor_class()),
            ]
        )
        return pipeline

    def train_and_save_model(
        self, pipeline, param_dist, model_name="SomeModel", output_path="model.pkl"
    ):
        """
        Utility function to run RandomizedSearchCV, fit a pipeline, log with MLflow,
        and then save the best model.
        """
        # MLflow setup
        mlflow.set_tracking_uri("http://localhost:8080")

        # Create or set the MLflow experiment
        experiment_name = "Days_Experiment"
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # Start MLflow run
        with mlflow.start_run(run_name=model_name) as run:
            # Log the parameters dictionary as a baseline reference
            mlflow.log_params({"search_space": str(param_dist)})

            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=20,
                cv=5,
                scoring="r2",
                random_state=42,
                verbose=1,
            )

            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_

            # Debug print: Check the type of best_model
            print("Type of best_model for", model_name, ":", type(best_model))
            
            # Evaluate on test set
            y_pred = best_model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # Print metrics to console
            print(f"Metrics for {model_name}:")
            print(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")

            # Infer model signature and log
            signature = infer_signature(self.X_train, best_model.predict(self.X_train))
            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=signature,
                input_example=self.X_train.iloc[:5],  # sample inputs
            )

            # Ensure the models directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save the model locally
            joblib.dump(best_model, output_path)
            print(f"Model {model_name} saved to {output_path}")

            # Print the MLflow run URL for reference
            run_id = mlflow.active_run().info.run_id
            print(
                f"MLflow Run URL: http://localhost/#/experiments/"
                f"{mlflow.active_run().info.experiment_id}/runs/{run_id}"
            )

        return best_model

    def train(self):
        """
        Main method to train both a Ridge model (v1) and a Linear Regression model (v2),
        then save them as separate pickle files.
        """
        # 1) Prepare dataset
        self.prepare_dataset()

        # 2) Build a pipeline for Ridge
        pipeline_ridge = self.build_pipeline(Ridge)
        # Define hyperparameters for the Ridge RandomizedSearchCV
        param_dist_ridge = {
            "regressor__alpha": np.logspace(-6, 3, 100),
            "preprocessor__num__imputer__strategy": ["mean", "median", "most_frequent"],
        }
        # Train and save the best Ridge model
        self.best_model_v1 = self.train_and_save_model(
            pipeline=pipeline_ridge,
            param_dist=param_dist_ridge,
            model_name="Ridge_Regression_v1",
            output_path=self.model_output_path_v1,
        )

        # 3) Build a pipeline for Linear Regression (v2)
        pipeline_linreg = self.build_pipeline(LinearRegression)
        # Define hyperparameters for the Linear Regression RandomizedSearchCV;
        # since LinearRegression has no regularization parameter by default,
        # we only tune the imputer strategy.
        param_dist_linreg = {
            "preprocessor__num__imputer__strategy": ["mean", "median", "most_frequent"],
        }
        # Train and save the best Linear Regression model
        self.best_model_v2 = self.train_and_save_model(
            pipeline=pipeline_linreg,
            param_dist=param_dist_linreg,
            model_name="Linear_Regression_v2",
            output_path=self.model_output_path_v2,
        )

        # 4) Save test datasets for reference or future predictions
        os.makedirs(self.test_cases_path, exist_ok=True)
        self.X_test.to_csv(os.path.join(self.test_cases_path, "X_test.csv"), index=False)
        self.y_test.to_csv(os.path.join(self.test_cases_path, "y_test.csv"), index=False)
        print(f"Test sets saved in {self.test_cases_path}")

        print("\nBoth models trained and saved successfully!")
        return self.best_model_v1, self.best_model_v2


if __name__ == "__main__":
    # Define paths (adjust as needed)
    PREPROCESSED_FILE = "Data/preprocessed/preprocessed_data.csv"

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Save each model under a different file name
    MODEL_PATH_V1 = "models/ridge_model_v1.pkl"
    MODEL_PATH_V2 = "models/linear_model_v2.pkl"

    TEST_CASES_FOLDER = "Test_Cases"

    trainer = ModelTrainer(
        preprocessed_file=PREPROCESSED_FILE,
        model_output_path_v1=MODEL_PATH_V1,
        model_output_path_v2=MODEL_PATH_V2,
        test_cases_path=TEST_CASES_FOLDER,
    )
    trainer.train()
