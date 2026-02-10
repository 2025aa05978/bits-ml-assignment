import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import os

# Page Config
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

# Title and Info
st.title("🏦 Loan Approval Prediction App (Live Training)")
st.markdown("""
This application demonstrates the full ML pipeline. 
It **trains the models on the fly** using your provided datasets and then allows for interactive prediction.
""")

# --- Backend: Data Loading & Model Training ---
@st.cache_resource
def train_and_get_models():
    """
    Loads data, cleans it, trains models, and returns resources.
    Cached so it only runs once.
    """
    status_text = st.empty()
    status_text.info("⏳ Training models... Please wait.")
    
    try:
        # 1. Load Datasets (Assuming files are in the same directory)
        # Check if files exist, else return None
        if not os.path.exists("Semi_Urban_loans.csv") or not os.path.exists("Rural_loans.csv"):
            return None, None, None, None, None

        semi_df = pd.read_csv("Semi_Urban_loans.csv")
        rural_df = pd.read_csv("Rural_loans.csv")
        
        # 2. Standardize & Merge
        rural_df = rural_df.rename(columns={
            'Loan_ID': 'LID', 'Sex': 'Gender', 'DependentsCount': 'Dependents',
            'Loan_Term': 'LoanAmountTerm', 'LoanAmount': 'LoanAmt',
            'Income': 'Applicant_Income', 'Loan_Status': 'Approved'
        })
        rural_df['Coapplicant_Income'] = 0.0
        rural_df['Credit_History'] = np.nan
        rural_df['Approved'] = rural_df['Approved'].map({'Yes': 1, 'No': 0})
        
        full_df = pd.concat([semi_df, rural_df], axis=0, ignore_index=True)
        full_df = full_df.drop(columns=['LID'])
        
        # 3. Cleaning
        full_df['Gender'] = full_df['Gender'].replace({1: 'Male', '1': 'Male', 'M': 'Male', 'F': 'Female', 0: 'Female'})
        full_df['Marrital_Status'] = full_df['Marrital_Status'].replace({'Yes': 1, 'No': 0, '1': 1, 1: 1, -1: np.nan, '-1': np.nan})
        full_df['Dependents'] = full_df['Dependents'].replace({'3+': 3}).astype(float)
        full_df['Education'] = full_df['Education'].replace({'G': 1, 'NG': 0, -1: np.nan, '-1': np.nan})
        full_df['Self_Employed'] = full_df['Self_Employed'].replace({'Yes': 1, 'No': 0, -1: np.nan})
        full_df['Credit_History'] = full_df['Credit_History'].replace({-1: np.nan, '-1': np.nan})
        
        num_cols = ['Applicant_Income', 'Coapplicant_Income', 'LoanAmt', 'LoanAmountTerm']
        for col in num_cols:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
            
        full_df = full_df.dropna(subset=['Approved'])
        full_df['Approved'] = full_df['Approved'].astype(int)
        
        # Imputation
        cat_cols = ['Gender', 'Marrital_Status', 'Dependents', 'Education', 'Self_Employed', 'Credit_History']
        imp_mode = SimpleImputer(strategy='most_frequent')
        full_df[cat_cols] = imp_mode.fit_transform(full_df[cat_cols])
        
        imp_median = SimpleImputer(strategy='median')
        full_df[num_cols] = imp_median.fit_transform(full_df[num_cols])
        
        # Encoding
        le = LabelEncoder()
        full_df['Gender'] = le.fit_transform(full_df['Gender'].astype(str))
        
        feature_columns = [c for c in full_df.columns if c != 'Approved']
        X = full_df.drop(columns=['Approved'])
        y = full_df['Approved']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            
        status_text.success("✅ Models trained successfully!")
        
        # Return test data for sample download
        test_df_sample = X_test.copy()
        test_df_sample['target'] = y_test
        
        return scaler, feature_columns, trained_models, test_df_sample, le

    except Exception as e:
        status_text.error(f"Training Failed: {e}")
        return None, None, None, None, None

# Run training
scaler, feature_cols, loaded_models, test_sample_df, gender_encoder = train_and_get_models()

if loaded_models is None:
    st.error("Could not load or train models. Ensure 'Rural_loans.csv' and 'Semi_Urban_loans.csv' are in the directory.")
    st.stop()

# --- Frontend: Sidebar & UI ---

st.sidebar.header("Configuration")

# 1. Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV (Test Data)", type=["csv"])

# 2. Model Selection
model_options = list(loaded_models.keys())
selected_model_name = st.sidebar.selectbox("Select Model", model_options)

# 3. Helper: Download Sample
st.sidebar.markdown("---")
st.sidebar.subheader("Don't have a file?")
if test_sample_df is not None:
    csv = test_sample_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "Download Sample Test Data",
        csv,
        "test_data_sample.csv",
        "text/csv",
        key='download-csv'
    )

# --- Main Logic ---

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### 📄 Uploaded Data Preview")
        st.dataframe(data.head())
        
        # Identify Target
        possible_targets = ['target', 'Approved', 'Loan_Status']
        target_col = next((col for col in possible_targets if col in data.columns), None)
        
        X_test = data.copy()
        y_test = None
        has_labels = False

        if target_col:
            y_test = data[target_col]
            X_test = data.drop(columns=[target_col])
            # Normalize target if it's Yes/No
            if y_test.dtype == 'object':
                 y_test = y_test.map({'Yes': 1, 'Y': 1, 'No': 0, 'N': 0})
            has_labels = True
        
        # --- Preprocessing for Inference ---
        # Map columns if user uploaded raw rural/semi data
        rename_map = {
            'Sex': 'Gender', 'DependentsCount': 'Dependents', 'Loan_Term': 'LoanAmountTerm',
            'LoanAmount': 'LoanAmt', 'Income': 'Applicant_Income'
        }
        X_test = X_test.rename(columns=rename_map)
        
        # Align columns
        for col in feature_cols:
            if col not in X_test.columns:
                X_test[col] = 0
                
        # Handle Data Types
        for col in X_test.columns:
             X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

        # Handle Gender Encoding if raw string passed (Simplified)
        # In a real app, use the fitted encoder properly. Here we assume numeric or clean input from sample.
        
        X_test = X_test[feature_cols]
        X_input = scaler.transform(X_test)

        # Prediction
        model = loaded_models[selected_model_name]
        preds = model.predict(X_input)
        probs = model.predict_proba(X_input)[:, 1]

        # Display Results
        st.divider()
        st.write(f"### 📊 Results: {selected_model_name}")
        
        col1, col2 = st.columns([1, 2])

        with col1:
            if has_labels:
                st.subheader("Metrics")
                acc = accuracy_score(y_test, preds)
                try:
                    auc = roc_auc_score(y_test, probs)
                except:
                    auc = 0.0
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1'],
                    'Value': [acc, auc, prec, rec, f1]
                })
                st.table(metrics_df.style.format({'Value': '{:.4f}'}))
            else:
                st.info("Upload data with labels to see metrics.")
                st.metric("Total Predictions", len(preds))

        with col2:
            if has_labels:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

        # Show predictions
        st.subheader("Prediction Details")
        
        # Color code predictions
        def highlight_pred(val):
            color = 'green' if val == 1 else 'red'
            return f'color: {color}'

        res_df = pd.DataFrame({
            'Loan_ID': data.index, 
            'Probability (Approval)': probs, 
            'Prediction': preds,
            'Prediction_Label': ['Approved' if p==1 else 'Rejected' for p in preds]
        })
        
        st.dataframe(res_df.style.map(highlight_pred, subset=['Prediction']), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.write("Ensure your columns match: Gender, Applicant_Income, LoanAmt, etc.")

else:
    st.info("👈 Please upload a CSV file or download the sample data from the sidebar.")
    st.markdown("#### Sample Data Preview (First 5 rows)")
    if test_sample_df is not None:
        st.dataframe(test_sample_df.head())