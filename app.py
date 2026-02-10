import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# Import model runners
from model.logistic_regression import run_logistic_regression
from model.decision_tree import run_decision_tree
from model.knn import run_knn
from model.naive_bayes import run_naive_bayes
from model.random_forest import run_random_forest
from model.xgboost import run_xgboost


st.set_page_config(page_title="ML Assignment 2", layout="centered")

st.title("Machine Learning Classification Models")
st.write("Adult Income Dataset – Model Comparison & Prediction")

if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------
# Download sample test dataset
# ---------------------------------------------------
st.subheader("Download Sample Test Dataset")

st.markdown(
    "[Click here to download test_data.csv](https://raw.githubusercontent.com/sweety-garai/ml-classification-models-streamlit/main/data/test_data.csv)"
)


# ---------------------------------------------------
# Upload CSV 
# ---------------------------------------------------
st.subheader("Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV file (same columns as training data)",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(uploaded_df.head())

# ---------------------------------------------------
# Model selection
# ---------------------------------------------------
st.subheader("Step 2: Select Model")

model_descriptions = {
    "Logistic Regression": "Linear baseline classifier.",
    "Decision Tree": "Rule-based non-linear classifier.",
    "kNN": "Distance-based classifier.",
    "Naive Bayes": "Probabilistic classifier.",
    "Random Forest": "Bagging-based ensemble model.",
    "XGBoost": "Boosting-based ensemble model."
}

model_choice = st.selectbox(
    "Choose a classification model",
    list(model_descriptions.keys())
)

st.info(model_descriptions[model_choice])

# ---------------------------------------------------
# Run Model
# ---------------------------------------------------
st.subheader("Step 3: Run Model")

if st.button("Run Model"):

    if model_choice == "Logistic Regression":
        st.session_state.results = run_logistic_regression()
    elif model_choice == "Decision Tree":
        st.session_state.results = run_decision_tree()
    elif model_choice == "kNN":
        st.session_state.results = run_knn()
    elif model_choice == "Naive Bayes":
        st.session_state.results = run_naive_bayes()
    elif model_choice == "Random Forest":
        st.session_state.results = run_random_forest()
    elif model_choice == "XGBoost":
        st.session_state.results = run_xgboost()

    st.success("Model trained successfully!")

# ---------------------------------------------------
# Evaluation Results
# ---------------------------------------------------
if st.session_state.results is not None:

    results = st.session_state.results

    st.subheader("Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [
            results["Accuracy"],
            results["AUC"],
            results["Precision"],
            results["Recall"],
            results["F1"],
            results["MCC"]
        ]
    })

    st.table(metrics_df)

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(results["y_test"], results["y_pred"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    # Class Distribution
    st.subheader("Class Distribution (Test Set)")
    st.bar_chart(results["y_test"].value_counts())

# ---------------------------------------------------
# Prediction on Uploaded Data
# ---------------------------------------------------
if uploaded_file is not None and st.session_state.results is not None:

    st.subheader("Prediction on Uploaded Data")

    try:
        uploaded_encoded = pd.get_dummies(uploaded_df, drop_first=True)

        for col in results["feature_columns"]:
            if col not in uploaded_encoded.columns:
                uploaded_encoded[col] = 0

        uploaded_encoded = uploaded_encoded[results["feature_columns"]]

        predictions = results["model_object"].predict(uploaded_encoded)
        uploaded_df["Predicted Income (>50K = 1)"] = predictions

        st.dataframe(uploaded_df)

    except Exception as e:
        st.error("Uploaded CSV format does not match training data.")
        st.error(str(e))
