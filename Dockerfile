FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .
RUN mkdir -p Data/raw/processed

ENV PYTHONPATH=/app
ENV FLASK_APP=predict_api.py

EXPOSE 9000


CMD ["python", "predict_api.py"]