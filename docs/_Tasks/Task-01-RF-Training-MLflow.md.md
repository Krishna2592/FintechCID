# Task 01: Random Forest Training with MLflow & Hyperparameter Tuning

## 🎯 Objective
Transform the raw Parquet data into a "Gold" model. We will train a deeply-tuned Random Forest classifier, track every experiment in MLflow, and save the best version for our LangGraph agents to use.

## 🏗️ Architecture Requirements

### 1. Scale Up
- Update `data/generate_transactions.py` to generate **1,000,000** records. 
- Ensure the fraud rate remains strictly at 5%.

### 2. Feature Engineering
- Create a training script `mlops/train_model.py`.
- Implement a simple feature engineering pipeline:
    - Hour of day (from timestamp).
    - Is Weekend (boolean).
    - Amount as a ratio of the user's average (if applicable) or simple scaling.
- Use `LabelEncoder` for categorical columns like `merchant_category`.

### 3. The MLflow Experiment
- Wrap the training in `with mlflow.start_run():`.
- **Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` on a 10% subset of the data to find the best `max_depth` and `n_estimators`.
- **Final Train:** Train the final model on the full 1M rows using the best parameters.
- **Metrics:** Log Precision, Recall (Focus!), F1-Score, and a Confusion Matrix plot to MLflow.

### 4. Artifact Preservation
- Save the trained `RandomForest` model and the `LabelEncoder` objects as MLflow artifacts.
- Register the model as "TransactionScreener_v1".

## 🚀 Execution Instructions for Claude Code
1. Increase data generation to 1M rows and re-run `generate_transactions.py`.
2. Write the `mlops/train_model.py` script.
3. Run the training script locally.
4. Verify that the MLflow UI (run `mlflow ui` in terminal) shows the logged parameters and metrics.