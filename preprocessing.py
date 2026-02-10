"""
This file contains the common preprocessing pipeline for the
Adult Income (Census Income) dataset.

It is imported by all model files to ensure:
- Same dataset
- Same preprocessing
- Same train-test split

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(test_size=0.2, random_state=42):
    """
    Loads and preprocesses the Adult Income dataset.

    Returns:
    X_train_scaled : np.ndarray
        Scaled training feature matrix
    X_test_scaled : np.ndarray
        Scaled test feature matrix
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels
    """

    # ---------------------------------------------------
    # Load the dataset
    # ---------------------------------------------------
    
    df = pd.read_csv("data/adult_income.csv")

    # ---------------------------------------------------
    # Handle missing values
    # ---------------------------------------------------
    # In this dataset, missing values are marked as '?'
    df.replace("?", np.nan, inplace=True)

    # Drop rows with missing values
    
    df.dropna(inplace=True)

    # ---------------------------------------------------
    # Encode the target variable
    # ---------------------------------------------------
    # income >50K  -> 1
    # income <=50K -> 0
    df["income"] = df["income"].apply(
        lambda x: 1 if x.strip() == ">50K" else 0
    )

    # ---------------------------------------------------
    # Separate features and target
    # ---------------------------------------------------
    X = df.drop("income", axis=1)
    y = df["income"]

    # ---------------------------------------------------
    # Encode categorical features
    # ---------------------------------------------------
    # Convert categorical columns to numeric using One-Hot Encoding
   
    X = pd.get_dummies(X, drop_first=True)

    # ---------------------------------------------------
    # Train-test split
    # ---------------------------------------------------
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ---------------------------------------------------
    # Feature scaling
    # ---------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    return X_train, X_test, y_train, y_test, X_train.columns
