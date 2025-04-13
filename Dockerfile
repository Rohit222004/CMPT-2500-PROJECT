# Dockerfile - for Flask + training container

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create folders and set permissions
RUN mkdir -p /app/Data/raw/processed /app/logs \
    && chmod -R 777 /app/logs

ENV PYTHONPATH=/app
ENV FLASK_APP=predict_api.py
ENV LOG_DIR=/app/logs

# Expose ports for Flask API and Prometheus metrics
EXPOSE 9000
EXPOSE 8002

# Only launch Flask; train.py will be called manually
CMD ["python", "predict_api.py"]
