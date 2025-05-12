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

File Structure



File/Directory
Description



new.csv
Input HPLC data (required columns: injection_time, column_serial_number).


streamlit_app.py
Streamlit dashboard for visualization.


anomaly_detection_with_oneclass_svm.py
Trains models and generates predictions.


train_model/
Output directory for models and predictions.


train_model/{param}_oneclass_svm_model.pkl
One-Class SVM models.


train_model/{param}_lstm_model.h5
LSTM models.


train_model/predictions3.csv
Predicted anomalies.


one_class_svm_model.pkl
Model for Streamlit visualization.


requirements.txt
Python dependencies.


README.md
Project documentation.


Setup Instructions
Prerequisites

Python 3.8+
Git
OpenAI API key (for GPT-4)
Libraries: pandas, numpy, scikit-learn, tensorflow, streamlit, plotly, joblib, openai, python-dotenv

Installation

Clone the Repository:
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Set Up OpenAI API Key:

Create a .env file:echo "OPENAI_API_KEY=your-api-key-here" > .env




Prepare Input Data:

Place new.csv in the project root.
Required columns: injection_time (datetime), column_serial_number.
Optional columns: peak_width_5, retention_time, amount_percent, etc.



Usage

Train Models and Generate Predictions:

Run the training script:python anomaly_detection_with_oneclass_svm.py


Output: Models and predictions3.csv in train_model/.


Launch the Streamlit Dashboard:

Start the Streamlit app:streamlit run streamlit_app.py


Open http://localhost:8501 in a browser.


Interact with the Dashboard:

Use sidebar filters: System Name, Method Set, Column Serial Number, Performance Metric.
View scatter plots (anomalies: orange/red X), deviation graphs, and tables.
Check GPT-4 summaries for future anomaly causes.



Input/Output
Input

File: new.csv
Format: CSV with columns like:injection_time,system_name,column_serial_number,peak_width_5,amount_percent
2024-10-05 10:00:00,Sys1,Col123,0.5,5.8



Output

File: train_model/predictions3.csv
Columns: predicted_date, predicted_{param}, anomaly_flag, anomaly_cause, anomaly_deviation, etc.
Example:predicted_date,predicted_amount_percent,anomaly_flag,anomaly_cause
2024-10-19,8.9,1,"High retention time; clean system"




Dashboard: Visualizes anomalies, deviations, and GPT-4 summaries.

Contributing

Fork the repository.
Create a branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add your feature".
Push: git push origin feature/your-feature.
Open a pull request.

License
MIT License. See LICENSE for details.
Contact
For support, open an issue on GitHub.
