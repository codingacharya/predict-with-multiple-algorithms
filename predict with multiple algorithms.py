import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Streamlit app title
st.title("Predict with Multiple ML Algorithms")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    
    # Select target variable
    target_column = st.selectbox("Select Target Column", df.columns)
    feature_columns = st.multiselect("Select Feature Columns", df.columns, default=[col for col in df.columns if col != target_column])
    
    if target_column and feature_columns:
        X = df[feature_columns]
        y = df[target_column]
        
        # Encoding categorical target variable if necessary
        if y.dtype == 'O':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Model selection
        classifier_name = st.selectbox("Choose Classifier", ["Random Forest", "SVM", "Logistic Regression", "XGBoost"])
        
        if st.button("Train and Predict"):
            if classifier_name == "Random Forest":
                model = RandomForestClassifier()
            elif classifier_name == "SVM":
                model = SVC()
            elif classifier_name == "Logistic Regression":
                model = LogisticRegression()
            elif classifier_name == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
                
            # Train the model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")
