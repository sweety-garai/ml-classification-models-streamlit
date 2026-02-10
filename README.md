# 📘 Machine Learning Assignment 2  
## Income Prediction Using Classification Models & Streamlit Deployment

---

## **Problem Statement**

In real-world socio-economic and policy-making scenarios, understanding the factors that influence an individual’s income level is extremely important.  
The objective of this project is to build a machine learning system that can **predict whether a person earns more than \$50,000 per year** based on demographic, educational, and employment-related attributes.

Multiple **classification models** are trained and evaluated on the same dataset to study how different algorithms perform on the income prediction task. The results are presented through an **interactive Streamlit web application**, which also allows users to upload new data and obtain income predictions using trained models.

---

## **Dataset Description**

The dataset used for this project is the **Adult Income Dataset**, a widely used benchmark dataset for income classification problems.

**Prediction Target**
- Whether an individual’s annual income is **greater than \$50K** or **less than or equal to \$50K**

**Dataset Characteristics**
- Combination of **categorical** (education, occupation, workclass, marital status, etc.) and **numerical** (age, hours-per-week, capital-gain, capital-loss) features  
- Binary target variable (`income`)  
- Real-world and slightly imbalanced dataset  
- Suitable for evaluating linear, non-linear, probabilistic, and ensemble models  

**Preprocessing Steps**
- One-hot encoding for categorical variables  
- Train–test split for evaluation  
- Same preprocessing pipeline applied across all models to ensure fair comparison  

---

## **Models Used**

The following classification models were implemented and evaluated on the same dataset:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (kNN)  
- Naive Bayes (Gaussian)  
- Random Forest (Ensemble Model)  
- XGBoost (Ensemble Model)  

All models were implemented using standard machine learning libraries such as **scikit-learn** and **xgboost**.

---

## **Model Comparison**

All evaluation metrics were calculated using the **internal test set** created during preprocessing.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8548 | 0.9132 | 0.7504 | 0.6245 | 0.6817 | 0.5928 |
| Decision Tree | 0.8583 | 0.8990 | 0.7870 | 0.5905 | 0.6748 | 0.5964 |
| kNN | 0.8195 | 0.8480 | 0.6504 | 0.5945 | 0.6212 | 0.5039 |
| Naive Bayes | 0.5488 | 0.7761 | 0.3514 | 0.9601 | 0.5144 | 0.3454 |
| Random Forest (Ensemble) | 0.8579 | 0.9120 | 0.7529 | 0.6391 | 0.6914 | 0.6033 |
| XGBoost (Ensemble) | **0.8754** | **0.9347** | **0.7957** | **0.6718** | **0.7285** | **0.6523** |

---

## **Observations on Model Performance**

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Served as a strong baseline model with balanced precision and recall, suitable for linear decision boundaries. |
| Decision Tree | Effectively captured non-linear relationships but showed slightly lower recall, indicating sensitivity to data splits. |
| kNN | Performance was moderate due to high dimensionality after one-hot encoding, which affects distance-based models. |
| Naive Bayes | Achieved very high recall but low precision, leading to many false positives and reduced overall effectiveness. |
| Random Forest (Ensemble) | Improved generalization compared to a single decision tree and produced stable, well-balanced metrics. |
| XGBoost (Ensemble) | Delivered the best overall performance across all metrics due to boosting and regularization, making it the most effective model for income prediction. |

---

## **Streamlit Application**

The Streamlit web application provides the following features:

- CSV upload option for test data  
- Download link for a sample test dataset  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  
- Income prediction on uploaded test data  

The application can be run locally using:

```bash
streamlit run app.py
