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
from sklearn.linear_model import Ridge
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

class ModelTrainer:
    def __init__(self, preprocessed_file: str, model_output_path: str, test_cases_path: str):
        """
        Initialize with the path to the preprocessed CSV file, model save path, and test cases folder.
        """
        self.preprocessed_file = preprocessed_file
        self.model_output_path = model_output_path
        self.test_cases_path = test_cases_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None

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
        X = df[['number_price_changes', 'vehicle_age', 'mileage', 'price', 'msrp',
                'dealer_name', 'listing_type', 'make', 'model', 'year', 'month', 'day']]
        y = df['days_on_market']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data split into training and test sets.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_pipeline(self):
        """
        Build the machine learning pipeline.
        """
        numerical_features = ['mileage', 'price', 'msrp', 'vehicle_age', 'year', 'month', 'day', 'number_price_changes']
        categorical_features = ['dealer_name', 'listing_type', 'make', 'model']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer()),  # Strategy will be tuned via hyperparameter search
                    ('scaler', StandardScaler()),
                ]), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        pipeline_Ridge = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Ridge(solver='lsqr'))
        ])
        return pipeline_Ridge

    def train(self):
        """
        Train the model using RandomizedSearchCV and save the best model.
        """
        self.prepare_dataset()
        pipeline_Ridge = self.build_pipeline()

        # Define hyperparameters for RandomizedSearchCV
        param_dist = {
            'regressor__alpha': np.logspace(-6, 3, 100),
            'preprocessor__num__imputer__strategy': ['mean', 'median', 'most_frequent']
        }

        # Set the MLflow tracking URI to the local server
        mlflow.set_tracking_uri("http://localhost:8080")

        # Create or set the MLflow experiment
        experiment_name = "Go_Auto_Days_on_market"
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # Start MLflow run
        with mlflow.start_run(run_name="Ridge_Regression") as run:
            # Log parameters
            mlflow.log_params(param_dist)

            random_search = RandomizedSearchCV(
                estimator=pipeline_Ridge,
                param_distributions=param_dist,
                n_iter=20,
                cv=5,
                scoring='r2',
                random_state=42
            )

            random_search.fit(self.X_train, self.y_train)
            self.best_model = random_search.best_estimator_

            # Log metrics
            y_pred = self.best_model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            # Infer model signature and log the model with signature and input example
            signature = infer_signature(self.X_train, self.best_model.predict(self.X_train))
            mlflow.sklearn.log_model(
                self.best_model,
                "model",
                signature=signature,
                input_example=self.X_train.iloc[:5]  # Example input (first 5 rows)
            )

            # Save the model locally
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(self.best_model, self.model_output_path)
            print(f"Model saved to {self.model_output_path}")

            # Save test datasets
            os.makedirs(self.test_cases_path, exist_ok=True)
            self.X_test.to_csv(os.path.join(self.test_cases_path, 'X_test.csv'), index=False)
            self.y_test.to_csv(os.path.join(self.test_cases_path, 'y_test.csv'), index=False)
            print(f"Test sets saved in {self.test_cases_path}")

            # Print the MLflow run URL
            print(f"MLflow Run URL: http://localhost/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}")

        return self.best_model

if __name__ == "__main__":
    # Define paths
    PREPROCESSED_FILE = 'Data/preprocessed/preprocessed_data.csv'
    MODEL_PATH = 'models/ridge_model.pkl'
    TEST_CASES_FOLDER = 'Test_Cases/'

    trainer = ModelTrainer(PREPROCESSED_FILE, MODEL_PATH, TEST_CASES_FOLDER)
    trainer.train()