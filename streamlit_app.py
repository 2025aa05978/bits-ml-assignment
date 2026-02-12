import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Page Config
st.set_page_config(page_title="Student Dropout Prediction", layout="wide", page_icon="🎓")

# App Header
st.title("Predict Students Dropout and Academic Success")
st.markdown("""
This application utilizes machine learning to predict whether a student will Dropout, remain Enrolled, or Graduate.
Upload a CSV dataset with student demographics and academic data to see model predictions and evaluations.
""")

# Sidebar Configuration
st.sidebar.header("Settings")

# Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload Student Data (CSV)", type=["csv"])
st.sidebar.info("Upload 'test_data.csv'.")

# Model Selection
model_options = [
    "Logistic Regression", 
    "Decision Tree", 
    "KNN", 
    "Naive Bayes", 
    "Random Forest", 
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select Classification Model", model_options)

# Load Pre-trained Resources
@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load('model/scaler.pkl')
        le = joblib.load('model/label_encoder.pkl')
        feature_cols = joblib.load('model/feature_columns.pkl')
        models = {}
        for name in model_options:
            filename = f"model/{name.replace(' ', '_').lower()}.pkl"
            if os.path.exists(filename):
                models[name] = joblib.load(filename)
        return scaler, le, feature_cols, models
    except Exception as e:
        return None, None, None, None

scaler, le, feature_cols, loaded_models = load_resources()

if not loaded_models:
    st.error("Error: Trained models not found.")
    st.stop()

# Main Logic & Inference
if uploaded_file is not None:
    try:
        # Read uploaded CSV (Handling both semicolon and comma separated values)
        data = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.write("### Uploaded Data Preview (First 5 Rows)")
        st.dataframe(data.head())
        
        # Check for target column to calculate metrics
        target_col = 'Target'
        has_labels = target_col in data.columns
        
        X_test = data.copy()
        y_test_encoded = None

        if has_labels:
            y_test_raw = data[target_col]
            X_test = data.drop(columns=[target_col])
            
            # Encode string targets to match model outputs (0, 1, 2)
            y_test_encoded = le.transform(y_test_raw.astype(str))
            
        # Ensure uploaded data has correct columns (filling missing with 0 for robustness)
        for col in feature_cols:
            if col not in X_test.columns:
                X_test[col] = 0
                
        # Reorder to match training precisely
        X_test = X_test[feature_cols]
        
        # Scale Features
        X_input_scaled = scaler.transform(X_test)

        # Make Predictions
        model = loaded_models[selected_model_name]
        preds = model.predict(X_input_scaled)
        probs = model.predict_proba(X_input_scaled)

        # Decode predictions back to original labels (Dropout, Enrolled, Graduate)
        decoded_preds = le.inverse_transform(preds)

        st.divider()
        st.write(f"### Model Performance: **{selected_model_name}**")
        
        col1, col2 = st.columns([1, 1.5])

        with col1:
            if has_labels:
                st.subheader("Evaluation Metrics")
                
                # Multi-class metrics using 'macro' average
                acc = accuracy_score(y_test_encoded, preds)
                auc = roc_auc_score(y_test_encoded, probs, multi_class='ovr', average='macro')
                prec = precision_score(y_test_encoded, preds, average='macro', zero_division=0)
                rec = recall_score(y_test_encoded, preds, average='macro', zero_division=0)
                f1 = f1_score(y_test_encoded, preds, average='macro', zero_division=0)
                mcc = matthews_corrcoef(y_test_encoded, preds)
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'ROC AUC (OVR)', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)', 'MCC'],
                    'Value': [acc, auc, prec, rec, f1, mcc]
                })
                st.table(metrics_df.style.format({'Value': '{:.4f}'}))
            else:
                st.info("Upload data containing the 'Target' column to view evaluation metrics.")
                st.metric("Total Records Predicted", len(preds))

        with col2:
            if has_labels:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test_encoded, preds)
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Use label encoder classes for ticks
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                            xticklabels=le.classes_, yticklabels=le.classes_)
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
                st.pyplot(fig)

        # Show detailed predictions
        st.subheader("Prediction Details")
        res_df = pd.DataFrame({
            'Record_Index': data.index,
            'Predicted_Status': decoded_preds
        })
        
        # Include original target if available for comparison
        if has_labels:
            res_df.insert(1, 'Actual_Status', data[target_col])
            
        # Add probability confidences
        for i, class_name in enumerate(le.classes_):
            res_df[f'Prob_{class_name}'] = np.round(probs[:, i], 4)
            
        st.dataframe(res_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.write("Please ensure the uploaded CSV matches the training data format (semicolon separated recommended).")

else:
    st.info("Please upload a CSV file to view predictions.")