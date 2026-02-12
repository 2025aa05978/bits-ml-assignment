import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef


# Load Data
print("Loading dataset for Predict Students Dropout and Academic Success")
try:
    df = pd.read_csv('data.csv', sep=';')
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the dataset is in the same directory.")
    exit()

# Preprocess Data
print(f"Dataset Shape: {df.shape}")
X = df.drop(columns=['Target'])
y = df['Target']

# Encode the categorical target ('Dropout', 'Enrolled', 'Graduate') to numeric (0, 1, 2)
# This is especially required for XGBoost
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'model/label_encoder.pkl')

print(f"Classes found: {le.classes_}")

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Save a sample of the test set for the Streamlit App to use
test_data = X_test.copy()
# Save original string labels for the test file
test_data['Target'] = le.inverse_transform(y_test)
test_data.to_csv("test_data_sample.csv", sep=';', index=False)
print("Saved 'test_data_sample.csv' for Streamlit App upload.")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(list(X.columns), 'model/feature_columns.pkl')

# Initialize Classification Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, multi_class='multinomial'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Train and Evaluate Models
print("\n" + "="*90)
print(f"{'Model':<22} | {'Accuracy':<8} | {'AUC (OVR)':<9} | {'Precision':<9} | {'Recall':<8} | {'F1 Score':<8} | {'MCC':<8}")
print("="*90)

for name, model in models.items():
    # Train Model
    model.fit(X_train_scaled, y_train)

    # Predict
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled) # For multi-class AUC

    # Calculate Metrics
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, preds)

    # Save Model
    file_name = f'model/{name.replace(" ", "_").lower()}.pkl'
    joblib.dump(model, file_name)

    # Print Metrics
    print(f"{name:<22} | {acc:.4f}   | {auc:.4f}     | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {mcc:.4f}")

print("="*90)
print("\nAll models successfully saved in the model directory.")