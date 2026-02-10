"""
Implements XGBoost Classifier on the Adult Income dataset.

"""

from preprocessing import get_data

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def run_xgboost():
    # ---------------------------------------------------
    # Load preprocessed data
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test, feature_columns = get_data()

    # ---------------------------------------------------
    # Initialize XGBoost model
    # ---------------------------------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

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
    print("\nXGBoost Performance")
    print("----------------------------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")

    return {
        "Model": "XGBoost",
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
    run_xgboost()
