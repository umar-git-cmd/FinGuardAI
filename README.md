# FinGuard AI

> High-Throughput Real-Time FinTech Risk Engine & Transaction Audit System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI 0.142](https://img.shields.io/badge/FastAPI-0.142.2-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit 1.64](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io)

FinGuard AI — высокопроизводительная система на базе машинного обучения для оценки рисков и выявления мошеннических финансовых транзакций с задержкой инференса менее 50 мс.

Проект разработан для **Global Innovation Build Challenge (GIBC) V2 — Track 02: Applied FinTech & MedTech**.

---

## Architecture

```text
[ Incoming Transaction ]
          |
          v
+---------------------+
|     FastAPI API     |
|                     |
| Pydantic Validation |
+---------------------+
          |
          v
+---------------------+
|      ML Engine      |
|                     |
| Fraud Classification|
+---------------------+
          |
          v
+---------------------+
|     Risk Engine     |
|                     |
| Approve             |
| 3D-Secure Review    |
| Block               |
+---------------------+
          |
          v
+---------------------+
|    Streamlit UI     |
|                     |
| Risk Monitoring     |
| Transaction Audit   |
+---------------------+





Key Features
Low-latency inference — designed for transaction risk scoring with inference latency below 50 ms.
Fraud detection — machine learning classification for identifying potentially fraudulent transactions.
Imbalanced data handling — designed for datasets where fraudulent transactions represent a small fraction of total transactions.
Separated architecture — FastAPI backend and Streamlit frontend are deployed as independent components.
Dynamic risk decisions — transactions can be classified into different risk categories based on model probability and configured thresholds.
Batch processing — supports processing multiple transactions in a single request.
CSV analysis — allows users to upload and analyze transaction datasets.
Model information endpoint — exposes model configuration, features, threshold, and batch limits.
Health monitoring — provides an API health-check endpoint.
Technology Stack
Component	Technology
Backend	FastAPI
Frontend	Streamlit
Machine Learning	Scikit-learn
Data Processing	Pandas
Model Serialization	Joblib
Validation	Pydantic
Server	Uvicorn
Language	Python 3.12
Project Structure
finguard-ai/
│
├── main.py
├── app.py
├── requirements.txt
├── README.md
│
└── finscam/
    └── ml_model/
        └── fraud_model.joblib
Backend

main.py contains the FastAPI application, model loading, validation, prediction endpoints, batch processing, and CSV processing.

Frontend

app.py contains the Streamlit interface for interacting with the fraud detection API.

Machine Learning Model

The trained model is stored in:

finscam/ml_model/fraud_model.joblib

The model bundle contains:

trained classification model
fraud detection threshold
feature list
API Endpoints
Method	Endpoint	Description
GET	/health	API health check
GET	/model-info	Model configuration and features
POST	/predict	Predict a single transaction
POST	/predict/batch	Predict multiple transactions
POST	/predict/csv	Analyze a CSV file

Interactive API documentation is available through Swagger:

http://localhost:8000/docs
Installation
1. Clone the repository
git clone https://github.com/<your-username>/finguard-ai.git
cd finguard-ai
2. Create a virtual environment
Windows
python -m venv venv
venv\Scripts\activate
macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Running the Application

The project consists of two services: the FastAPI backend and the Streamlit frontend.

Start FastAPI
uvicorn main:app --reload --port 8000

The API will be available at:

http://localhost:8000

Swagger documentation:

http://localhost:8000/docs
Start Streamlit

Open a second terminal and run:

streamlit run app.py

The Streamlit interface will be available at:

http://localhost:8501
Configuration

The backend supports environment variables for model configuration and batch size.

Model path
MODEL_PATH=/path/to/fraud_model.joblib
Maximum batch size
MAX_BATCH=10000

If no environment variables are provided, the application uses the default configuration defined in main.py.

Model Workflow

The prediction pipeline works as follows:

Transaction
     |
     v
Input Validation
     |
     v
Feature Selection
     |
     v
ML Model
     |
     v
Fraud Probability
     |
     v
Threshold Comparison
     |
     +-------------------+
     |                   |
     v                   v
Legitimate             Fraud

A transaction is classified as fraudulent when:

probability >= threshold

The threshold is loaded from the trained model bundle and can be inspected through:

GET /model-info
Dataset and Metrics

The project uses the European credit card transaction benchmark dataset containing 284,807 transactions.

Current project targets/metrics:

Metric	Target
PR-AUC (AUPRC)	> 0.86
ROC-AUC	> 0.98
Recall	> 88%
Fraud rate	< 0.17%

Metrics should be updated if the final trained model produces different validated results.

Usage
Single Transaction

The Streamlit interface allows users to:

Enter transaction features.
Submit the transaction for analysis.
Receive the fraud probability.
Compare the probability against the configured threshold.
View the final classification.
CSV Analysis

Users can upload a CSV file containing the required model features.

The application validates:

required columns
numeric values
empty files
maximum number of rows

After processing, the interface displays prediction results and fraud statistics.

API Example
Request
POST /predict
Content-Type: application/json

Example:

{
  "feature_1": 0.12,
  "feature_2": -1.42,
  "feature_3": 0.87
}

The exact feature names depend on the trained model and can be retrieved from:

GET /model-info
Response
{
  "probability": 0.923451,
  "is_fraud": true,
  "threshold": 0.5
}
Error Handling

The API validates incoming data and returns appropriate HTTP errors for:

missing features
invalid numeric values
empty batch requests
excessive batch size
invalid CSV files
missing CSV columns
empty CSV files
NaN or non-numeric values

The Streamlit interface converts API errors into user-readable messages.

Performance

The system is designed for low-latency transaction scoring and lightweight deployment.

The target inference latency is:

< 50 ms

Actual latency depends on hardware, network overhead, request size, model complexity, and deployment configuration.

Team
Member	Responsibility
Isa Zholdoshbekov (@isko6709)	ML Pipeline & Imbalanced Data Modeling
Bayell Ruslanov (@Bayell)	API Architecture & Backend
Kutman Nurkalykov (@kutman09)	Streamlit UI/UX & Visualization
Umar Dev (@umardev808)	Data Validation & System Testing
License

This project was developed as part of the Global Innovation Build Challenge (GIBC) V2.
