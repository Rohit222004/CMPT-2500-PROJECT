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

# Import logging setup
from logging_config import configure_logging

# Initialize logger for training
loggers = configure_logging()
logger = loggers['train']


class ModelTrainer:
    def __init__(
        self,
        preprocessed_file: str,
        model_output_path_v1: str,
        model_output_path_v2: str,
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

        self.best_model_v1 = None
        self.best_model_v2 = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the preprocessed CSV data.
        """
        logger.info(f"📥 Loading data from {self.preprocessed_file}")
        df = pd.read_csv(self.preprocessed_file)
        logger.info(f"✅ Data loaded successfully with shape {df.shape}")
        return df

    def prepare_dataset(self):
        """
        Prepare the dataset by splitting into features and target, then train/test split.
        """
        df = self.load_data()
        logger.info("🔎 Preparing dataset...")

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

        logger.info(f"✅ Data split into training and test sets:")
        logger.info(f"   - Training set: {self.X_train.shape}")
        logger.info(f"   - Test set: {self.X_test.shape}")

    def build_pipeline(self, regressor_class):
        """
        Build a pipeline given a particular regressor class (e.g. Ridge or LinearRegression).
        """
        logger.info(f"🔨 Building pipeline with {regressor_class.__name__}")

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
                            ("imputer", SimpleImputer(strategy="mean")),
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

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor_class()),
            ]
        )

        logger.info(f"✅ Pipeline created successfully!")
        return pipeline

    def train_and_save_model(self, pipeline, param_dist, model_name, output_path):
        """
        Train model, evaluate, log with MLflow, and save.
        """
        logger.info(f"🚀 Training model: {model_name}")

        # ✅ MLflow setup
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # ✅ Create or get experiment
        experiment_name = "Days_Experiment"
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            logger.info(f"📌 Creating experiment '{experiment_name}'")
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"📌 Using existing experiment ID: {experiment_id}")

        mlflow.set_experiment(experiment_name)

        # ✅ Start MLflow run
        with mlflow.start_run(run_name=model_name) as run:
            logger.info(f"✅ MLflow run started with ID: {run.info.run_id}")

            mlflow.log_params({"search_space": str(param_dist)})

            logger.info("🔎 Running RandomizedSearchCV...")

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

            logger.info("✅ Model training completed")

            # ✅ Make predictions and log metrics
            y_pred = best_model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"✅ {model_name} Metrics - MSE: {mse}, RMSE: {rmse}, R2: {r2}")

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # ✅ Log model
            signature = infer_signature(self.X_train, best_model.predict(self.X_train))
            mlflow.sklearn.log_model(
                best_model, "model", signature=signature, input_example=self.X_train.iloc[:5]
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            joblib.dump(best_model, output_path)

            logger.info(f"✅ {model_name} saved to {output_path}")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"🔗 MLflow Run URL: http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}")

        # ✅ Ensure the run is properly closed
        mlflow.end_run()

    def train(self):
        """
        Main method to train both models.
        """
        self.prepare_dataset()

        # ✅ Ridge Regression
        pipeline_ridge = self.build_pipeline(Ridge)
        param_dist_ridge = {
            "regressor__alpha": np.logspace(-6, 3, 100),
            "preprocessor__num__imputer__strategy": ["mean", "median", "most_frequent"],
        }
        self.best_model_v1 = self.train_and_save_model(
            pipeline_ridge, param_dist_ridge, "Ridge_Regression_v1", self.model_output_path_v1
        )

        # ✅ Linear Regression
        pipeline_linreg = self.build_pipeline(LinearRegression)
        param_dist_linreg = {
            "preprocessor__num__imputer__strategy": ["mean", "median", "most_frequent"],
        }
        self.best_model_v2 = self.train_and_save_model(
            pipeline_linreg, param_dist_linreg, "Linear_Regression_v2", self.model_output_path_v2
        )


if __name__ == "__main__":
    trainer = ModelTrainer(
        preprocessed_file="Data/preprocessed/preprocessed_data.csv",
        model_output_path_v1="models/ridge_model_v1.pkl",
        model_output_path_v2="models/linear_model_v2.pkl",
        test_cases_path="Test_Cases"
    )
    trainer.train()
