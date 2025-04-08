# utils/monitoring.py

from prometheus_client import start_http_server, Gauge
import threading
import time

class TrainingMonitor:
    def __init__(self, port=8002):
        self.port = port
        self._start_server()

    def _start_server(self):
        """Start a Prometheus metrics server in a background thread"""
        thread = threading.Thread(target=start_http_server, args=(self.port,), daemon=True)
        thread.start()
        print(f"✅ Prometheus metrics server started at http://localhost:{self.port}/metrics")


class RegressionMonitor(TrainingMonitor):
    def __init__(self, port=8002):
        super().__init__(port)

        # Regression-specific metrics
        self.mse = Gauge('regression_mean_squared_error', 'Mean Squared Error')
        self.rmse = Gauge('regression_root_mean_squared_error', 'Root Mean Squared Error')
        self.mae = Gauge('regression_mean_absolute_error', 'Mean Absolute Error')
        self.r_squared = Gauge('regression_r_squared', 'R-squared coefficient')

        # Feature importance tracking (top 5 features)
        self.feature_importance = Gauge('feature_importance', 'Feature importance value', ['feature_name'])

    def record_metrics(self, mse=None, rmse=None, mae=None, r_squared=None, feature_importance=None):
        """Record regression metrics"""
        if mse is not None:
            self.mse.set(mse)
        if rmse is not None:
            self.rmse.set(rmse)
        if mae is not None:
            self.mae.set(mae)
        if r_squared is not None:
            self.r_squared.set(r_squared)

        # Update feature importance for top features
        if feature_importance is not None:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature_name, importance in sorted_features:
                self.feature_importance.labels(feature_name=feature_name).set(importance)
