Chromatography Anomaly Detection System
This project provides a machine learning solution for detecting and predicting anomalies in High-Performance Liquid Chromatography (HPLC) systems. It combines One-Class SVM for historical anomaly detection, LSTM for future predictions, a Streamlit dashboard for visualization, and GPT-4 for actionable insights. The system helps ensure HPLC reliability by identifying issues like column clogging or contamination early.
Features

Historical Anomaly Detection: Uses One-Class SVM to flag outliers in HPLC data with IQR-based deviation thresholds.
Future Predictions: Forecasts HPLC parameters (e.g., retention time, peak width) for 14 days using LSTM.
Interactive Dashboard: Streamlit UI with filters, scatter plots, deviation graphs, and data tables.
Actionable Insights: GPT-4 suggests chromatography-specific causes (e.g., "High peak width due to column clogging").
Robust Preprocessing: Handles missing data, encodes categoricals, and adds features like injection count.

How It Works

Data Input: Loads new.csv with HPLC data (e.g., injection_time, amount_percent).
Preprocessing:
Converts timestamps, fills missing values, clips outliers.
Adds features: injection_count, days_since_start.
Encodes categorical columns (e.g., system_name).


Anomaly Detection:
One-Class SVM (nu=0.1) detects historical anomalies using IQR thresholds (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR).
LSTM predicts future values; SVM flags anomalies.


Visualization: Streamlit dashboard shows plots, tables, and GPT-4 summaries.
Output: Saves predictions to train_model/predictions3.csv.


MIT License. See LICENSE for details.
Contact
For support, open an issue on GitHub.
