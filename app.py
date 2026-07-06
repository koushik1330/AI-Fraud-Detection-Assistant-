import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "notebook" / "data"
IMAGE_DIR = BASE_DIR / "notebook" / "images"
MODEL_DIR = BASE_DIR / "models"

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_DIR / "xgboost.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_model()

# --------------------------------------------------
# Load Test Dataset
# --------------------------------------------------

@st.cache_data
def load_test_data():
    try:
        X_test = joblib.load(DATA_DIR / "X_test.pkl")
        y_test = joblib.load(DATA_DIR / "y_test.pkl")
        return X_test, y_test
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None, None

X_test, y_test = load_test_data()
if X_test is None or y_test is None:
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 EDA Dashboard",
        "📈 Model Performance",
        "💳 Fraud Prediction"
    ]
)

# --------------------------------------------------
# HOME
# --------------------------------------------------

if page == "🏠 Home":

    st.title("💳 Credit Card Fraud Detection")

    st.markdown("""
    ### Project Overview

    This project predicts fraudulent credit card transactions using Machine Learning.

    The application demonstrates:

    - Data Analysis
    - Exploratory Data Analysis
    - Machine Learning
    - Explainable AI
    - Fraud Prediction
    """)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Transactions","284,807")
    c2.metric("Fraud","492")
    c3.metric("Fraud Rate","0.172%")
    c4.metric("Best Model","XGBoost")

    st.divider()

    st.subheader("Technologies")

    st.write("""
    - Python
    - Streamlit
    - XGBoost
    - Scikit-Learn
    - SHAP
    - Groq
    """)

# --------------------------------------------------
# EDA
# --------------------------------------------------

elif page == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Class Distribution",
            "Distributions",
            "Feature Correlation",
            "SMOTE Comparison"
        ]
    )

    with tab1:

        st.image(
            IMAGE_DIR / "class_distribution.png",
            use_container_width=True
        )

    with tab2:

        st.image(
            IMAGE_DIR / "distributions.png",
            use_container_width=True
        )

    with tab3:

        st.image(
            IMAGE_DIR/"feature_correlation.png",
            use_container_width=True
        )

    with tab4:

        st.image(
            IMAGE_DIR/"smote_comparison.png",
            use_container_width=True
        )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric("Accuracy","99.95%")
    c2.metric("Precision","94.2%")
    c3.metric("Recall","91.5%")
    c4.metric("F1","92.8%")
    c5.metric("ROC-AUC","0.998")

    st.divider()

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({

        "Model":[
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],

        "Accuracy":[
            0.98,
            0.999,
            0.9995
        ]

    })

    st.dataframe(comparison,use_container_width=True)

    st.subheader("Confusion Matrix")

    st.image(
    IMAGE_DIR / "final_confusion_matrix.png",
    use_container_width=True
    )

    st.subheader("ROC Curve")

    st.image(
    IMAGE_DIR / "model_performance_curves.png",
    use_container_width=True
    )

# --------------------------------------------------
# FRAUD PREDICTION
# --------------------------------------------------

elif page == "💳 Fraud Prediction":

    st.title("💳 Fraud Prediction")

    st.write("Select a transaction from the unseen test dataset.")

    index = st.selectbox(
        "Select Test Transaction",
        X_test.index
    )

    sample = X_test.loc[[index]]

    actual = int(y_test.loc[index])

    st.subheader("Selected Transaction")

    transaction = sample.T
    transaction.columns = ["Value"]

    st.dataframe(
        transaction,
        use_container_width=True
    )

    if st.button("Predict"):
        if model is None:
            st.error("Model could not be loaded.")
            st.stop()

        prediction = model.predict(sample)[0]

        probability = model.predict_proba(sample)[0][1]

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Actual Class",
                "Fraud 🚨" if actual==1 else "Legitimate ✅"
            )

        with col2:

            st.metric(
                "Predicted",
                "Fraud 🚨" if prediction==1 else "Legitimate ✅"
            )

        with col3:

            st.metric(
                "Fraud Probability",
                f"{probability*100:.2f}%"
            )

        if prediction==actual:

            st.success("✅ Correct Prediction")

        else:

            st.error("❌ Incorrect Prediction")

        st.divider()

        st.subheader("Prediction Summary")

        if prediction==1:

            st.error(
                f"""
                The model predicts this transaction is FRAUD.

                Confidence : {probability*100:.2f}%
                """
            )

        else:

            st.success(
                f"""
                The model predicts this transaction is LEGITIMATE.

                Confidence : {(1-probability)*100:.2f}%
                """
            )

        st.info(
            """
            In the next version this section will include:

            • SHAP Feature Importance

            • Waterfall Plot

            • AI Explanation (Groq)

            • Download Prediction Report
            """
        )
