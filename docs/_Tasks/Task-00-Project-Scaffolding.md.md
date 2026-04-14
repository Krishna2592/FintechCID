# Task 00: Project Scaffolding & Data Generation

## 🎯 Objective
Initialize the repository structure for "FintechCID," a DevSecOps Multi-Agent Financial Fraud Detection System. Set up the Python environment dependencies and build the foundational PySpark script to generate synthetic, enterprise-grade financial transaction data.

## 🏗️ Architecture Requirements

### 1. Directory Structure
Create the following modular directory structure:
- `/agents` (For LangGraph state and node definitions)
- `/api` (For the FastAPI gateway and DevSecOps models)
- `/core_logic` (For standard Python utilities and PageIndex retrieval)
- `/data/raw` (For generated synthetic data)
- `/mlops` (For MLflow tracking and Scikit-Learn training scripts)
- `/frontend` (For the Streamlit auditor dashboard)

### 2. Dependency Management
Create a `requirements.txt` file with the following core libraries pinned to current stable versions:
- `langgraph`, `langchain`, `langchain-community`
- `fastapi`, `uvicorn`, `pydantic`
- `mlflow`, `scikit-learn`, `pandas`, `numpy`
- `pyspark`
- `streamlit`

### 3. The Data Engine (PySpark)
Write a Python script at `/data/generate_transactions.py`.
- **Engine:** Use PySpark to simulate a scalable data lake ingestion process.
- **Task:** Generate a DataFrame of 10,000 synthetic transaction records.
- **Schema:** Include the following columns:
  - `transaction_id` (UUID)
  - `user_id` (String)
  - `timestamp` (Datetime)
  - `amount` (Float)
  - `merchant_category` (String)
  - `distance_from_home_km` (Float)
  - `velocity_24h` (Integer - number of transactions by user in last 24h)
  - `is_fraud` (Integer: 0 or 1. Force a strict 5% fraud rate).
- **Fraud Logic:** Inject logical patterns for fraud (e.g., high `velocity_24h` combined with large `distance_from_home_km` and high `amount` should have a higher probability of `is_fraud = 1`).
- **Output:** The script must save the final dataset as a parquet file in `/data/raw/transactions.parquet`.

## 🚀 Execution Instructions for Claude Code
1. Read this specification completely.
2. Execute the shell commands to create the directory structure.
3. Write the `requirements.txt` file.
4. Write the `/data/generate_transactions.py` script with robust comments.
5. Execute the PySpark script to verify it successfully writes the parquet file.
6. Report back when the data generation is complete.