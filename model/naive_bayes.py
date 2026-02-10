"""
Implements Gaussian Naive Bayes on the Adult Income dataset.

"""

from preprocessing import get_data

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def run_naive_bayes():
    # ---------------------------------------------------
    # Load preprocessed data
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test, feature_columns = get_data()

    # ---------------------------------------------------
    # Initialize Gaussian Naive Bayes model
    # ---------------------------------------------------
    model = GaussianNB()

    # ---------------------------------------------------
    # Train the model
    # ---------------------------------------------------
    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Predictions
    # ---------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------
    # Evaluation metrics (ALL REQUIRED)
    # ---------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # ---------------------------------------------------
    # Print results
    # ---------------------------------------------------
    print("\nNaive Bayes (Gaussian) Performance")
    print("----------------------------------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")

    return {
        "Model": "Naive Bayes",
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "y_test": y_test,
        "y_pred": y_pred,
        "model_object": model,
        "feature_columns": feature_columns
    }

if __name__ == "__main__":
    run_naive_bayes()
