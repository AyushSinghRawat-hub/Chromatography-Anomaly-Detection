import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import joblib
from datetime import datetime, timedelta
import warnings
import os
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings('ignore')

CSV_FILE = "data.csv"
MODEL_OUTPUT = "oneclass_svm_model.pkl"
LSTM_MODEL_OUTPUT = "lstm_model.h5"
PREDICTIONS_OUTPUT = "predictions.csv"
RANDOM_STATE = 42
NU = 0.1
INJECTION_THRESHOLD = 1000
SEQ_LENGTH = 10
N_FUTURE = 14

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{file_path}' not found.")

    expected_columns = [
        'injection_time', 'system_name', 'column_serial_number', 'peak_width_5', 'retention_time', 
        'signal_to_noise_ratio', 'amount_percent', 'amount_value', 'area_percent', 'area_value', 
        'peak_width_50', 'resolution', 'analyte', 'method_set_name', 'project', 'sample_name', 
        'system_operator'
    ]
    available_columns = df.columns.tolist()
    missing_columns = [col for col in expected_columns if col not in available_columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}. Using available columns: {available_columns}")

    if len(df.columns) != len(df.columns.unique()):
        print("Warning: Duplicate column names detected in CSV. Removing duplicates...")
        df = df.loc[:, ~df.columns.duplicated()]

    if 'injection_time' not in df.columns:
        raise ValueError("Required column 'injection_time' is missing.")

    df['injection_time'] = pd.to_datetime(df['injection_time'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['column_serial_number', 'injection_time'])

    numeric_cols = [
        'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 'amount_value', 
        'area_percent', 'area_value', 'peak_width_50', 'resolution', 'peak_width_10'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df.groupby('column_serial_number')[col].transform(
                lambda x: x.fillna(x.median()) if x.median() == x.median() else x.fillna(0)
            )
            df[col] = df[col].clip(lower=0, upper=df[col].quantile(0.99))

    df = df.sort_values(['column_serial_number', 'injection_time'])
    df['injection_count'] = df.groupby('column_serial_number').cumcount() + 1
    df['days_since_start'] = (df['injection_time'] - df.groupby('column_serial_number')['injection_time'].transform('min')).dt.days

    categorical_cols = [
        'system_name', 'column_serial_number', 'analyte', 'method_set_name', 'project', 
        'sample_name', 'system_operator',
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_original'] = df[col]

    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders

def prepare_features(df, selected_param, for_lstm=False):
    if for_lstm:
        for lag in range(1, 3):
            df[f'{selected_param}_lag_{lag}'] = df.groupby('column_serial_number')[selected_param].shift(lag)

    feature_cols = [
        selected_param, 'injection_count', 'days_since_start',
        'resolution', 'retention_time', 'peak_width_5', 'peak_width_50',
        'system_name', 'analyte',
        f'{selected_param}_lag_1', f'{selected_param}_lag_2'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    scaler = MinMaxScaler() if for_lstm else StandardScaler()
    X_scaled = scaler.fit_transform(X)

    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(df[[selected_param]])

    return X_scaled, scaler, feature_cols, target_scaler

def train_anomaly_model(X):
    model = OneClassSVM(kernel='rbf', nu=NU, gamma='scale')
    model.fit(X)
    return model

def detect_anomalies(df, X, model, feature_cols, selected_param):
    df['anomaly'] = model.predict(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    df['anomaly_score'] = model.decision_function(X)
    
    mean_val = df[selected_param].mean()
    std_val = df[selected_param].std()
    upper_threshold = mean_val + 3 * std_val
    lower_threshold = mean_val - 3 * std_val
    
    df['anomaly_feature'] = selected_param
    df['anomaly_deviation'] = df[selected_param].apply(
        lambda x: x - upper_threshold if x > upper_threshold else 
                  lower_threshold - x if x < lower_threshold else 0
    )
    return df

def create_sequences(data, seq_length, target_cols, feature_cols):
    X, y = [], []
    for col_serial in data['column_serial_number'].unique():
        col_data = data[data['column_serial_number'] == col_serial].sort_values('injection_time')
        if len(col_data) >= seq_length:
            for i in range(len(col_data) - seq_length):
                seq_X = col_data[feature_cols].iloc[i:i+seq_length].values
                seq_y = col_data[target_cols].iloc[i+seq_length].values
                if np.any(np.isnan(seq_X)) or np.any(np.isnan(seq_y)):
                    continue
                X.append(seq_X)
                y.append(seq_y)
    return np.array(X) if X else None, np.array(y) if y else None

def build_lstm_model(input_shape):
    timesteps, features = input_shape
    model = Sequential([
        Bidirectional(LSTM(150, return_sequences=True, input_shape=(timesteps, features))),
        Dropout(0.3),
        Bidirectional(LSTM(150, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(150)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=Huber())
    return model

def predict_future(model, last_sequence, n_future, target_cols, feature_cols, target_scaler, historical_stats):
    predictions = []
    current_seq = last_sequence.copy()
    
    valid_target_cols = [col for col in target_cols if col in feature_cols]
    if not valid_target_cols:
        raise ValueError("No valid target columns found in feature_cols.")
    
    hist_mean, hist_std = historical_stats['mean'], historical_stats['std']
    
    expected_shape = (SEQ_LENGTH, len(feature_cols))
    if last_sequence.shape != expected_shape:
        print(f"Warning: Invalid last_sequence shape: {last_sequence.shape}, expected: {expected_shape}. Skipping predictions.")
        return np.array([]), valid_target_cols
    if np.any(np.isnan(last_sequence)):
        print("Warning: NaN values detected in last_sequence. Skipping predictions.")
        return np.array([]), valid_target_cols
    
    for _ in range(n_future):
        x_input = current_seq.reshape((1, SEQ_LENGTH, len(feature_cols)))
        try:
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred[0, 0])
        except Exception as e:
            print(f"Error during model prediction: {str(e)}. Skipping this prediction step.")
            return np.array(predictions) if predictions else np.array([]), valid_target_cols
        
        new_row = np.zeros((1, len(feature_cols)))
        target_indices = [feature_cols.index(col) for col in valid_target_cols]
        new_row[0, target_indices] = pred[0, 0]
        
        if 'injection_count' in feature_cols:
            new_row[0, feature_cols.index('injection_count')] = current_seq[-1, feature_cols.index('injection_count')] + 1
        if 'days_since_start' in feature_cols:
            new_row[0, feature_cols.index('days_since_start')] = current_seq[-1, feature_cols.index('days_since_start')] + 1
        
        for col in ['resolution', 'retention_time', 'peak_width_5', 'peak_width_50']:
            if f'{col}_std' in historical_stats:
                idx = feature_cols.index(col)
                last_value = current_seq[-1, idx]
                col_std = historical_stats.get(f'{col}_std', 0.01)
                new_row[0, idx] = last_value + np.random.normal(0, col_std * 0.1)
        
        current_seq = np.vstack((current_seq[1:], new_row))
    
    predictions = np.array(predictions).reshape(-1, 1)
    unscaled_predictions = target_scaler.inverse_transform(predictions)
    
    pred_mean = unscaled_predictions.mean()
    pred_std = unscaled_predictions.std()
    if pred_std > 0:
        normalized_predictions = (unscaled_predictions - pred_mean) / pred_std * hist_std + hist_mean
    else:
        normalized_predictions = unscaled_predictions
    
    hist_min, hist_max = historical_stats['min'], historical_stats['max']
    normalized_predictions = np.clip(normalized_predictions, hist_min, hist_max)
    
    return normalized_predictions, valid_target_cols

def assign_anomaly_cause(row, selected_param, is_future_prediction=False, historical_stats=None):
    param_key = f'predicted_{selected_param}' if is_future_prediction else selected_param
    if param_key not in row:
        return "Unknown"
    
    value = row[param_key]
    anomaly_flag = row.get('anomaly_flag', False) if is_future_prediction else row.get('anomaly', 0) == 1
    anomaly_score = row.get('anomaly_score', 0)
    deviation = row.get('anomaly_deviation', 0)

    if not anomaly_flag and deviation == 0:
        return "No anomaly detected"

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if is_future_prediction and historical_stats:
        mean_val = historical_stats['mean']
        std_val = historical_stats['std']
    else:
        mean_val = row.get(f'{selected_param}_mean', value)
        std_val = row.get(f'{selected_param}_std', 0)
    
    upper_threshold = mean_val + 3 * std_val
    lower_threshold = mean_val - 3 * std_val
    
    injection_count = row.get('injection_count', 0)
    system_name = row.get('system_name_original', 'Unknown')
    analyte = row.get('analyte_original', 'Unknown')
    column_serial = row.get('column_serial_number_original', 'Unknown')
    method_set = row.get('method_set_name_original', 'Unknown')
    sample_name = row.get('sample_name_original', 'Unknown')

    prompt = (
        f"You are an expert in high-performance liquid chromatography (HPLC) analyzing anomalies in a chromatography system. "
        f"An anomaly was detected for the parameter '{selected_param}' with a value of {value:.3f}. "
        f"Statistical context: historical mean is {mean_val:.3f}, standard deviation is {std_val:.3f}, "
        f"upper threshold is {upper_threshold:.3f} (mean + 3*std), lower threshold is {lower_threshold:.3f} (mean - 3*std). "
        f"The anomaly deviation is {deviation:.3f} from the threshold, indicating {'high' if deviation > 0 else 'low'} deviation. "
        f"Anomaly score is {anomaly_score:.3f} (lower scores indicate stronger anomalies). "
        f"Contextual data: injection count is {injection_count} (threshold for overuse: {INJECTION_THRESHOLD}), "
        f"system name is '{system_name}', analyte is '{analyte}', column serial number is '{column_serial}', "
        f"method set is '{method_set}', sample name is '{sample_name}'. "
        f"Task: Provide a detailed, chromatography-specific cause for this anomaly, including: "
        f"- The parameter and its deviation (e.g., 'High peak_width_5 by 0.023'). "
        f"- The likely cause, mapped to HPLC issues: "
        f"  - High peak_width_5/peak_width_50: column clogging, degradation, or improper packing. "
        f"  - High retention_time: sample contamination, stationary phase deterioration, or mobile phase composition errors. "
        f"  - Low signal_to_noise_ratio: detector noise, dirty flow cell, or misalignment. "
        f"  - High injection_count (>1000): column overuse contributing to performance issues. "
        f"- An actionable recommendation (e.g., 'Clean tubing', 'Replace column'). "
        f"Format the response concisely, e.g., 'High peak_width_5 (deviation: 0.023) due to column clogging; recommend column cleaning or replacement.' "
        f"Assume an anomaly exists if deviation is non-zero or anomaly score is negative."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a chromatography expert identifying causes of anomalies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating cause: {str(e)}"

def predict_future_anomalies(df, lstm_model, target_cols, feature_cols, target_scaler, threshold, prediction_start, selected_param):
    predictions = []
    color_by_cols = ['system_name', 'analyte', 'method_set_name', 'project', 'sample_name', 'system_operator']
    
    historical_stats = {
        'mean': df[target_cols[0]].mean(),
        'std': df[target_cols[0]].std(),
        'min': df[target_cols[0]].min(),
        'max': df[target_cols[0]].max()
    }
    for col in ['resolution', 'retention_time', 'peak_width_5', 'peak_width_50']:
        if col in df.columns:
            historical_stats[f'{col}_std'] = df[col].std()
    
    for col_serial in df['column_serial_number'].unique():
        col_data = df[df['column_serial_number'] == col_serial].sort_values('injection_time')
        if len(col_data) < SEQ_LENGTH:
            print(f"Column {col_serial} has only {len(col_data)} rows, less than SEQ_LENGTH ({SEQ_LENGTH}). Skipping.")
            continue
                
        last_sequence = col_data[feature_cols].iloc[-SEQ_LENGTH:].values
        if np.any(np.isnan(last_sequence)):
            print(f"NaN values in last_sequence for column {col_serial}. Skipping.")
            continue
                
        last_date = col_data['injection_time'].iloc[-1]
        days_to_start = (prediction_start - last_date).days
        if days_to_start < 0:
            print(f"Prediction start date {prediction_start} is before the last historical date {last_date}. Starting predictions from {last_date + timedelta(days=1)}.")
            days_to_start = 1
            
        future_dates = [prediction_start + timedelta(days=i) for i in range(N_FUTURE)]
        injection_count = col_data['injection_count'].iloc[-1] + days_to_start
            
        future_preds, valid_target_cols = predict_future(
            lstm_model, last_sequence, N_FUTURE, target_cols, feature_cols, target_scaler, historical_stats
        )
            
        pred_df = pd.DataFrame({
            'column_serial_number': col_serial,
            'column_serial_number_original': col_data['column_serial_number_original'].iloc[-1],
            'predicted_date': future_dates,
            'injection_count': injection_count + np.arange(1, N_FUTURE + 1),
            'parameter': selected_param
        })
            
        for col in color_by_cols:
            if col in col_data.columns:
                pred_df[col] = col_data[col].iloc[-1]
                if f'{col}_original' in col_data.columns:
                    pred_df[f'{col}_original'] = col_data[f'{col}_original'].iloc[-1]
            else:
                pred_df[col] = 'Unknown'
                pred_df[f'{col}_original'] = 'Unknown'
            
        for i, col in enumerate(valid_target_cols):
            pred_df[f'predicted_{col}'] = future_preds[:, i] if future_preds.size > 0 else np.nan
            
        if future_preds.size > 0:
            pred_features = [f'predicted_{col}' for col in valid_target_cols]
            if not pred_features:
                print(f"No valid prediction features for column {col_serial}. Skipping anomaly detection.")
                continue
                
            svm_model = OneClassSVM(kernel='rbf', nu=NU, gamma='scale')
            X_pred = pred_df[pred_features].values
            if np.any(np.isnan(X_pred)):
                print(f"NaN values in prediction features for column {col_serial}. Skipping anomaly detection.")
                continue
                
            anomaly_scores = svm_model.fit_predict(X_pred)
            pred_df['anomaly_flag'] = anomaly_scores == -1
            pred_df['anomaly_score'] = svm_model.decision_function(X_pred)
            pred_df['replacement_alert'] = pred_df['injection_count'] > INJECTION_THRESHOLD
                
            mean_val = historical_stats['mean']
            std_val = historical_stats['std']
            upper_threshold = mean_val + 3 * std_val
            lower_threshold = mean_val - 3 * std_val
                
            pred_df['anomaly_feature'] = valid_target_cols[0]
            pred_df['anomaly_deviation'] = pred_df[f'predicted_{valid_target_cols[0]}'].apply(
                lambda x: x - upper_threshold if x > upper_threshold else 
                          lower_threshold - x if x < lower_threshold else 0
            )
                
            print(f"Column {col_serial}: {pred_df['anomaly_flag'].sum()} anomalies detected")
            print(f"Debug: Future predictions for {selected_param} (first 5 rows):\n{pred_df.head()}")
                
            pred_df['anomaly_cause'] = pred_df.apply(
                lambda row: assign_anomaly_cause(row, valid_target_cols[0], is_future_prediction=True, historical_stats=historical_stats), axis=1
            )
            predictions.append(pred_df)
        else:
            print(f"No future predictions generated for column {col_serial}, parameter {selected_param}.")
    
    return pd.concat(predictions) if predictions else pd.DataFrame()

def main():
    df, label_encoders = load_and_preprocess_data(CSV_FILE)
    
    available_params = [
        col for col in [
            'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 
            'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 
            'peak_width_10'
        ] if col in df.columns
    ]
    
    start_date = pd.Timestamp('2024-10-05').tz_localize(None)
    end_date = pd.Timestamp('2024-10-18').tz_localize(None)
    prediction_start = end_date + timedelta(days=1)
    
    filtered_data = df[(df['injection_time'] >= start_date) & (df['injection_time'] <= end_date)]
    if filtered_data.empty:
        print("Error: No data found for the selected date range.")
        return
    
    all_predictions = []
    
    for selected_param in available_params:
        print(f"Processing parameter: {selected_param}")
        
        X, scaler, feature_cols, target_scaler = prepare_features(filtered_data, selected_param)
        svm_model = train_anomaly_model(X)
        filtered_data_param = detect_anomalies(filtered_data.copy(), X, svm_model, feature_cols, selected_param)
        
        joblib.dump({'model': svm_model, 'scaler': scaler, 'label_encoders': label_encoders}, f"{selected_param}_{MODEL_OUTPUT}")
        
        filtered_data_param['anomaly_cause'] = filtered_data_param.apply(
            lambda row: assign_anomaly_cause(row, selected_param, is_future_prediction=False), axis=1
        )
        
        historical_data = filtered_data_param[[
            'injection_time', 'column_serial_number', 'column_serial_number_original', 
            selected_param, 'anomaly', 'anomaly_score', 'anomaly_feature', 'anomaly_deviation', 
            'anomaly_cause', 'injection_count', 'system_name', 'system_name_original', 
            'analyte', 'analyte_original', 'method_set_name', 'method_set_name_original', 
            'sample_name', 'sample_name_original', 'project', 'project_original', 
            'system_operator', 'system_operator_original'
        ]].copy()
        for col in available_params:
            if col != selected_param and col in filtered_data_param.columns:
                historical_data[col] = filtered_data_param[col]
            historical_data[f'predicted_{col}'] = np.nan
        historical_data['parameter'] = selected_param
        historical_data['predicted_date'] = historical_data['injection_time']
        historical_data['anomaly_flag'] = historical_data['anomaly'].astype(bool)
        historical_data['replacement_alert'] = historical_data['injection_count'] > INJECTION_THRESHOLD
        all_predictions.append(historical_data)
        
        X_lstm, lstm_scaler, lstm_feature_cols, lstm_target_scaler = prepare_features(filtered_data, selected_param, for_lstm=True)
        X_seq, y_seq = create_sequences(filtered_data, SEQ_LENGTH, [selected_param], lstm_feature_cols)
        
        if X_seq is not None and y_seq is not None and len(X_seq) > 0 and len(y_seq) > 0:
            train_size = int(0.8 * len(X_seq))
            X_train, X_val = X_seq[:train_size], X_seq[train_size:]
            y_train, y_val = y_seq[:train_size], y_seq[train_size:]
            if len(X_val) == 0 or len(y_val) == 0:
                print(f"Warning: Insufficient data for validation split for {selected_param}. Skipping LSTM training.")
                continue
            
            lstm_model = build_lstm_model((SEQ_LENGTH, len(lstm_feature_cols)))
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                          epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
            lstm_model.save(f"{selected_param}_{LSTM_MODEL_OUTPUT}")
            
            future_anomalies = predict_future_anomalies(
                filtered_data, lstm_model, [selected_param], lstm_feature_cols, lstm_target_scaler, 
                np.mean(filtered_data[selected_param]) + 3 * np.std(filtered_data[selected_param]), 
                prediction_start, selected_param
            )
            
            if not future_anomalies.empty:
                print(f"Debug: Future anomalies for {selected_param} (rows: {len(future_anomalies)})")
                for col in available_params:
                    if col != selected_param:
                        future_anomalies[f'predicted_{col}'] = np.nan
                all_predictions.append(future_anomalies)
        else:
            print(f"Warning: LSTM predictions skipped for {selected_param} due to insufficient sequence data.")
    
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        print("Debug: Final predictions shape:", final_predictions.shape)
        print("Debug: Final predictions columns:", final_predictions.columns.tolist())
        print("Debug: Sample predictions (first 5 rows):\n", final_predictions.head())
        print(f"Debug: Date range: {final_predictions['predicted_date'].min()} to {final_predictions['predicted_date'].max()}")
        final_predictions.to_csv(PREDICTIONS_OUTPUT, index=False)
        print(f"Predictions saved to {PREDICTIONS_OUTPUT}")
    else:
        print("Error: No predictions generated. Check data or model configuration.")

if __name__ == "__main__":
    main()