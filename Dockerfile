# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them early to leverage layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 

# Copy source code last to take advantage of Docker layer caching
COPY . .

# ✅ Create necessary directories with permissions
RUN mkdir -p /app/Data/raw/processed
RUN mkdir -p /app/logs
RUN chmod -R 777 /app/logs

# ✅ Environment variables for Flask and logging
ENV PYTHONPATH=/app
ENV FLASK_APP=predict_api.py
ENV LOG_DIR=/app/logs

# ✅ Expose Flask port
EXPOSE 9000

# ✅ Start Flask app
CMD ["python", "predict_api.py"]
