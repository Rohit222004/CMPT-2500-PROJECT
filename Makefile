# Project Variables
VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Paths
MODEL_PATH=models/ridge_model_v1.pkl
PREPROCESSED_DATA=Data/preprocessed/preprocessed_data.csv
TEST_CASES_DIR=Test_Cases/
X_TEST=$(TEST_CASES_DIR)/X_test.csv
Y_TEST=$(TEST_CASES_DIR)/y_test.csv
PREDICTIONS_FILE=$(TEST_CASES_DIR)/model_predictions_combined.csv

# Default target: Run full pipeline including API startup
all: init-cpu format lint test preprocess train run_api evaluate

# Create virtual environment and install dependencies (CPU version)
init-cpu:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	@echo "Installing additional requirements..."
	$(PIP) install -r requirements.txt
	@echo "Installing Pytorch - CPU"
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	@echo "Setup complete! Run 'source $(VENV)/bin/activate' to activate the environment."

# Format code using Black
format:
	@echo "Formatting code with Black..."
	$(PYTHON) -m black .

# Lint code using Flake8
lint:
	@echo "Checking code style with Flake8..."
	$(PYTHON) -m flake8 .

# Run unit tests
test:
	@echo "Running tests..."
	$(PYTHON) -m unittest discover -s tests -p "*.py"

# Preprocess data
preprocess:
	@echo "Running data preprocessing..."
	$(PYTHON) preprocess.py

# Train model
train:
	@echo "Training model..."
	$(PYTHON) train.py
	@echo "Model saved to $(MODEL_PATH)"

# Run the API
run_api:
	@echo "Starting Flask API..."
	$(PYTHON) src/predict_api.py

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	$(PYTHON) evaluate.py

# Clean temporary files
clean:
	@echo "Cleaning up temporary files..."
	rm -rf __pycache__ */__pycache__
	rm -rf $(VENV)
	rm -f $(MODEL_PATH)
	rm -f $(PREPROCESSED_DATA) $(X_TEST) $(Y_TEST) $(PREDICTIONS_FILE)

# Re-run full pipeline
run_all: preprocess train run_api evaluate

# Activate virtual environment
activate:
	@echo "Activating virtual environment..."
	. $(VENV)/bin/activate
