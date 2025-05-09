import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
import os

CSV_FILE = "data.csv"
PREDICTED_CSV_FILE = "predictions.csv"
MODEL_OUTPUT = "one_class_svm_model.pkl"
RANDOM_STATE = 42
NU = 0.1
INJECTION_THRESHOLD = 1000

@st.cache_data
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
        st.warning(f"Missing columns: {missing_columns}. Using available columns: {available_columns}")

    if len(df.columns) != len(df.columns.unique()):
        st.warning("Duplicate column names detected in CSV. Removing duplicates...")
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
        'sample_name', 'system_operator'
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

def prepare_features(df, selected_param):
    feature_cols = [
        selected_param, 'injection_count', 'days_since_start',
        'resolution', 'retention_time', 'peak_width_5', 'peak_width_50',
        'system_name', 'analyte'
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing feature columns: {missing_cols}")

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_cols

def train_anomaly_model(X):
    model = OneClassSVM(nu=NU, kernel='rbf', gamma='scale')
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
    
    print(f"Selected Parameter: {selected_param}")
    print(f"Mean: {mean_val:.3f}, Std: {std_val:.3f}")
    print(f"Thresholds: Lower = {lower_threshold:.3f}, Upper = {upper_threshold:.3f}")
    print(f"Data Range: Min = {df[selected_param].min():.3f}, Max = {df[selected_param].max():.3f}")
    print(f"Number of points outside thresholds: {((df[selected_param] < lower_threshold) | (df[selected_param] > upper_threshold)).sum()}")
    
    df['anomaly_feature'] = selected_param
    df['anomaly_deviation'] = df[selected_param].apply(
        lambda x: x - upper_threshold if x > upper_threshold else 
                  lower_threshold - x if x < lower_threshold else 0
    )
    
    return df, {'mean': mean_val, 'std': std_val, 'upper_threshold': upper_threshold, 'lower_threshold': lower_threshold}

def load_predicted_anomalies(file_path, selected_param, stats):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Predicted anomalies CSV file '{file_path}' not found.")
        return pd.DataFrame()

    print(f"Columns in {file_path}: {df.columns.tolist()}")

    if 'predicted_date' not in df.columns:
        st.error("Required column 'predicted_date' missing in predicted anomalies CSV.")
        return pd.DataFrame()

    df['predicted_date'] = pd.to_datetime(df['predicted_date'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['predicted_date'])

    required_cols = [
        f'predicted_{selected_param}', 'anomaly_flag', 'anomaly_cause', 
        'replacement_alert', 'column_serial_number', 'injection_count'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns in predicted anomalies CSV: {missing_cols}")
        if 'column_serial_number' not in df.columns:
            df['column_serial_number'] = 'Unknown'
            if 'column_serial_number_original' not in df.columns:
                df['column_serial_number_original'] = 'Unknown'

    if f'predicted_{selected_param}' in df.columns:
        upper_threshold = stats['upper_threshold']
        lower_threshold = stats['lower_threshold']
        df['anomaly_deviation'] = df[f'predicted_{selected_param}'].apply(
            lambda x: x - upper_threshold if x > upper_threshold else 
                      lower_threshold - x if x < lower_threshold else 0
        )
        
        print(f"Predicted Parameter: predicted_{selected_param}")
        print(f"Thresholds: Lower = {lower_threshold:.3f}, Upper = {upper_threshold:.3f}")
        print(f"Predicted Data Range: Min = {df[f'predicted_{selected_param}'].min():.3f}, Max = {df[f'predicted_{selected_param}'].max():.3f}")
        print(f"Number of predicted points outside thresholds: {((df[f'predicted_{selected_param}'] < lower_threshold) | (df[f'predicted_{selected_param}'] > upper_threshold)).sum()}")
    else:
        st.warning(f"Column 'predicted_{selected_param}' not found in predicted data. Setting anomaly_deviation to 0.")
        df['anomaly_deviation'] = 0

    df['anomaly_feature'] = selected_param

    return df

def wrap_text(text, width=30):
    return '<br>'.join(textwrap.wrap(text, width=width))

def main():
    st.set_page_config(layout="wide")

    st.title("Chromatography Anomaly Detection Dashboard")

    try:
        df, label_encoders = load_and_preprocess_data(CSV_FILE)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    st.sidebar.header("Apply Filters")
    start_date = datetime(2024, 10, 5)
    end_date = datetime(2024, 10, 18)
    system_name = st.sidebar.selectbox("System Name", ["All"] + sorted(df['system_name_original'].unique().tolist()))
    method_set = st.sidebar.selectbox("Method Set", ["All"] + sorted(df['method_set_name_original'].unique().tolist()))
    selected_column = st.sidebar.multiselect("Column Serial Number", ["All"] + sorted(df['column_serial_number_original'].unique().tolist()), default=["All"])
    show_predicted_table = st.sidebar.checkbox("Show Predicted Data Table", value=False)
    show_deviation_graph = st.sidebar.checkbox("Show Deviation Graph", value=False)
    deviation_time_range = st.sidebar.selectbox("Deviation Time Range", ["One Day", "One Week", "One Month"], index=2)
    color_by = st.sidebar.selectbox("Color By", [
        'system_name', 'analyte', 'method_set_name', 'project', 'sample_name', 
        'system_operator'
    ])

    available_params = [
        col for col in [
            'peak_width_5', 'retention_time', 'signal_to_noise_ratio', 'amount_percent', 
            'amount_value', 'area_percent', 'area_value', 'peak_width_50', 'resolution', 
            'peak_width_10'
        ] if col in df.columns
    ]
    selected_param = st.sidebar.selectbox("Performance Metric", available_params, index=0)

    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)
    prediction_start = end_date + timedelta(days=1)

    filtered_data = df[(df['injection_time'] >= start_date) & (df['injection_time'] <= end_date)]
    if system_name != "All":
        filtered_data = filtered_data[filtered_data['system_name_original'] == system_name]
    if method_set != "All":
        filtered_data = filtered_data[filtered_data['method_set_name_original'] == method_set]
    if "All" not in selected_column and selected_column:
        filtered_data = filtered_data[filtered_data['column_serial_number_original'].isin(selected_column)]

    if filtered_data.empty:
        st.error("No data found for the selected date range.")
        return

    filtered_data = filtered_data.loc[:, ~filtered_data.columns.duplicated()]

    try:
        X, scaler, feature_cols = prepare_features(filtered_data, selected_param)
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return

    svm_model = train_anomaly_model(X)
    filtered_data, stats = detect_anomalies(filtered_data, X, svm_model, feature_cols, selected_param)
    joblib.dump({'model': svm_model, 'scaler': scaler, 'label_encoders': label_encoders}, MODEL_OUTPUT)

    future_anomalies = load_predicted_anomalies(PREDICTED_CSV_FILE, selected_param, stats)

    fig = go.Figure()

    color_col = color_by + '_original' if f'{color_by}_original' in filtered_data.columns else color_by

    fig.add_trace(go.Scatter(
        x=filtered_data['injection_time'],
        y=filtered_data[selected_param],
        mode='markers',
        name='Historical Data',
        marker=dict(
            color=filtered_data[color_by],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_by)
        ),
        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{selected_param}</b>: %{{y:.3f}}<br><b>{color_by}</b>: %{{customdata}}<extra></extra>',
        customdata=filtered_data[color_col]
    ))

    historical_anomalies = filtered_data[filtered_data['anomaly'] == 1]
    print("Historical Anomalies Shape:", historical_anomalies.shape)
    print("Historical Anomalies Columns:", historical_anomalies.columns.tolist())
    print("Historical Anomalies Sample:", historical_anomalies[[color_col, 'anomaly_feature', 'anomaly_deviation', 'anomaly_score']].head(2).to_dict())
    print("Anomaly Deviation Values:", historical_anomalies['anomaly_deviation'].tolist())

    fig.add_trace(go.Scatter(
        x=historical_anomalies['injection_time'],
        y=historical_anomalies[selected_param],
        mode='markers',
        name='Historical Anomalies',
        marker=dict(color='orange', symbol='x'),
        text=historical_anomalies.apply(
            lambda row: (
                f"Date: {row['injection_time']}<br>"
                f"Parameter: {selected_param}: {row[selected_param]:.3f}<br>"
                f"Anomaly: Yes<br>"
                f"{color_by}: {str(row[color_col]) if pd.notnull(row[color_col]) else 'Unknown'}<br>"
                f"Feature: {str(row['anomaly_feature']) if pd.notnull(row['anomaly_feature']) else selected_param}<br>"
                f"Deviation: {f'{float(row['anomaly_deviation']):.3f}' if pd.notnull(row['anomaly_deviation']) else '0.000'}<br>"
                f"Severity: {'Severe' if pd.notnull(row['anomaly_score']) and row['anomaly_score'] < -0.05 else 'Moderate' if pd.notnull(row['anomaly_score']) and row['anomaly_score'] < 0 else 'Normal'}"
            ), axis=1
        ),
        hovertemplate='%{text}<extra></extra>',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            align='left'
        )
    ))

    if not future_anomalies.empty:
        pred_color_col = color_by + '_original' if f'{color_by}_original' in future_anomalies.columns else color_by
        print(f"pred_color_col: {pred_color_col}")

        fig.add_trace(go.Scatter(
            x=future_anomalies['predicted_date'],
            y=future_anomalies[f'predicted_{selected_param}'],
            mode='markers',
            name='Predicted Data',
            visible='legendonly',
            marker=dict(
                color=future_anomalies[color_by] if color_by in future_anomalies.columns else 'blue',
                colorscale='Viridis',
                showscale=False
            ),
            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>{selected_param}</b>: %{{y:.3f}}<br><b>{color_by}</b>: %{{customdata}}<extra></extra>',
            customdata=future_anomalies[pred_color_col] if pred_color_col in future_anomalies.columns else future_anomalies.get('column_serial_number', 'Unknown')
        ))

        predicted_anomalies = future_anomalies[future_anomalies['anomaly_flag']]
        if not predicted_anomalies.empty:
            try:
                customdata = np.array(predicted_anomalies[[pred_color_col, 'anomaly_feature', 'anomaly_deviation', 'anomaly_cause', 'anomaly_score']].apply(
                    lambda row: [
                        str(row[pred_color_col]) if pd.notnull(row[pred_color_col]) else str(row.get('column_serial_number_original', row.get('column_serial_number', 'Unknown'))),
                        str(row['anomaly_feature']) if pd.notnull(row['anomaly_feature']) else selected_param,
                        float(row['anomaly_deviation']) if pd.notnull(row['anomaly_deviation']) else 0.0,
                        wrap_text(str(row['anomaly_cause'])) if pd.notnull(row['anomaly_cause']) else 'Unknown',
                        '███' if pd.notnull(row['anomaly_score']) and row['anomaly_score'] < -0.05 else '█' if pd.notnull(row['anomaly_score']) and row['anomaly_score'] < 0 else ''
                    ], axis=1
                ).tolist(), dtype=object)
            except Exception as e:
                print(f"Error creating customdata for predicted anomalies: {e}")
                customdata = np.array([[
                    str(row[pred_color_col]) if pd.notnull(row[pred_color_col]) else 'Unknown',
                    selected_param,
                    0.0,
                    'Unknown',
                    ''
                ] for _, row in predicted_anomalies.iterrows()], dtype=object)

            fig.add_trace(go.Scatter(
                x=predicted_anomalies['predicted_date'],
                y=predicted_anomalies[f'predicted_{selected_param}'],
                mode='markers',
                name='Predicted Anomalies',
                marker=dict(color='red', symbol='x'),
                hovertemplate=(
                    f'<b>Date</b>: %{{x}}<br>'
                    f'<b>{selected_param}</b>: %{{y:.3f}}<br>'
                    f'<b>Anomaly</b>: Yes<br>'
                    f'<b>{color_by}</b>: %{{customdata[0]}}<br>'
                    f'<b>Feature</b>: %{{customdata[1]}}<br>'
                    f'<b>Deviation</b>: %{{customdata[2]:.3f}}<br>'
                    f'<b>Cause</b>: %{{customdata[3]}}<br>'
                    f'<b>Severity</b>: %{{customdata[4]}}<extra></extra>'
                ),
                customdata=customdata,
                hoverlabel=dict(
                    namelength=-1,
                    font_size=12,
                    align='left'
                )
            ))

    fig.update_layout(
        title=f'Predicted {selected_param} with Anomalies',
        xaxis_title='Injection Time',
        yaxis_title=selected_param,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0,
            orientation="h"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Historical Data Table")
    columns_to_display = list(dict.fromkeys([
        'injection_time', selected_param, 'resolution', 'retention_time', 
        'anomaly_feature', 'anomaly_deviation'
    ]))
    st.dataframe(filtered_data[columns_to_display])

    if show_predicted_table and not future_anomalies.empty:
        st.subheader("Predicted Data Table")
        columns_to_display_pred = list(dict.fromkeys([
            'predicted_date', f'predicted_{selected_param}', 'anomaly_flag', 
            'anomaly_cause', 'replacement_alert', 'anomaly_deviation'
        ]))
        columns_to_display_pred = [col for col in columns_to_display_pred if col in future_anomalies.columns]
        st.dataframe(future_anomalies[columns_to_display_pred])

    if show_deviation_graph:
        time_range_label = deviation_time_range
        if deviation_time_range == "One Day":
            time_delta = timedelta(days=1)
        elif deviation_time_range == "One Week":
            time_delta = timedelta(days=7)
        else:  # One Month
            time_delta = timedelta(days=30)

        latest_historical = filtered_data['injection_time'].max() if not filtered_data.empty else pd.Timestamp.now()
        latest_predicted = future_anomalies['predicted_date'].max() if not future_anomalies.empty else latest_historical
        latest_date = max(latest_historical, latest_predicted)

        deviation_data = filtered_data[
            (filtered_data['anomaly_deviation'] != 0) & 
            (filtered_data['injection_time'] >= latest_date - time_delta)
        ]
        pred_deviation_data = future_anomalies[
            (future_anomalies['anomaly_deviation'] != 0) & 
            (future_anomalies['predicted_date'] >= latest_date - time_delta)
        ] if not future_anomalies.empty else pd.DataFrame()

        st.subheader(f"Deviation of {selected_param} Over Time (Last {time_range_label}, Historical and Predicted)")
        
        fig_deviation = go.Figure()

        if not deviation_data.empty:
            fig_deviation.add_trace(go.Scatter(
                x=deviation_data['injection_time'],
                y=deviation_data['anomaly_deviation'],
                mode='markers',
                name='Historical Deviations',
                marker=dict(
                    color='blue',
                    size=8
                ),
                hovertemplate=(
                    f'<b>Date</b>: %{{x}}<br>'
                    f'<b>Deviation</b>: %{{y:.3f}}<extra></extra>'
                )
            ))

        if not pred_deviation_data.empty:
            fig_deviation.add_trace(go.Scatter(
                x=pred_deviation_data['predicted_date'],
                y=pred_deviation_data['anomaly_deviation'],
                mode='markers',
                name='Predicted Deviations',
                marker=dict(
                    color='red',
                    size=8
                ),
                hovertemplate=(
                    f'<b>Date</b>: %{{x}}<br>'
                    f'<b>Deviation</b>: %{{y:.3f}}<extra></extra>'
                )
            ))

        if not deviation_data.empty or not pred_deviation_data.empty:
            fig_deviation.update_layout(
                title=f'Deviation of {selected_param} Over Time (Last {time_range_label}, Historical and Predicted)',
                xaxis_title='Date',
                yaxis_title='Anomaly Deviation',
                hovermode='closest',
                showlegend=True
            )
            st.plotly_chart(fig_deviation, use_container_width=True)
        else:
            st.info(f"No deviations found for {selected_param} in the last {time_range_label}.")

    st.text(f"Showing historical data for {len(filtered_data)} injections")
    st.text(f"Showing predictions for {len(future_anomalies)} future dates starting from {prediction_start.strftime('%Y-%m-%d')}")

    if not future_anomalies.empty:
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        anomaly_summary = future_anomalies[future_anomalies['anomaly_flag']]
        if not anomaly_summary.empty:
            summary_text = f"Summary of future anomalies for {selected_param} starting from {prediction_start.strftime('%Y-%m-%d')}:\n"
            for index, row in anomaly_summary.iterrows():
                summary_text += (
                    f"- Date: {row['predicted_date']}, "
                    f"Predicted Value: {row[f'predicted_{selected_param}']:.3f}, "
                    f"Cause: {row['anomaly_cause']}, "
                    f"Replacement Alert: {row['replacement_alert']}\n"
                )
            summary_text += "Please review these predictions and take action if necessary."

            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a chromatography expert summarizing anomaly predictions. "
                                "Provide a concise summary in tabular format, including Date, Parameter, Cause, and Replacement Alert. "
                                "Ensure causes are specific to chromatography (e.g., column clogging for high peak width, contamination for high retention time) "
                                "and include actionable recommendations. "
                                "If no data like values are found, omit them (e.g., do not show NaN or empty fields)."
                            )
                        },
                        {"role": "user", "content": f"Summarize the following anomaly data: {summary_text}"}
                    ],
                    max_tokens=500
                )

                st.subheader("Future Anomaly Summary")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.warning(f"Error generating anomaly summary: {e}")
        else:
            st.warning("No future anomalies detected. Check model sensitivity or data range.")

if __name__ == "__main__":
    main()