# Bank Marketing Classification – Machine Learning Assignment 2

## 1. Project Overview

This project applies five supervised machine learning classification algorithms to predict whether a bank customer will subscribe to a term deposit.

The project includes model training, performance evaluation, model comparison, and an interactive Streamlit application. The application loads previously trained models and uses uploaded customer data to generate predictions.

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**File used:** `bank.csv`

Dataset details:

- Total records: 4,521
- Input features: 15
- Target variable: `y`
- Target classes: `yes` and `no`

The dataset contains customer demographic information, account details, and information about previous marketing campaigns.

The target variable indicates whether a customer subscribed to a term deposit.

- `yes`: Customer subscribed.
- `no`: Customer did not subscribe.

The original CSV file uses a semicolon (`;`) as its delimiter.

## 3. Machine Learning Algorithms

The following five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

All five models were trained using the same training and testing split.

## 4. Model Evaluation

The models were evaluated using six performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Area Under the ROC Curve (AUC)
- Matthews Correlation Coefficient (MCC)

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7050 | 0.2237 | 0.6346 | 0.3308 | 0.7350 | 0.2373 |
| Decision Tree | 0.7724 | 0.2287 | 0.4135 | 0.2945 | 0.6788 | 0.1827 |
| KNN | 0.8762 | 0.2778 | 0.0481 | 0.0820 | 0.6112 | 0.0727 |
| Naive Bayes | 0.8099 | 0.2763 | 0.4038 | 0.3281 | 0.7166 | 0.2274 |
| SVM | 0.7901 | 0.2913 | 0.5769 | 0.3871 | 0.7502 | 0.3002 |

### Observations

- KNN achieved the highest accuracy, but its recall for the positive class was low.
- SVM achieved the highest F1 Score, AUC, and MCC among the five models.
- Logistic Regression achieved the highest recall.
- The dataset has an imbalanced target distribution, so accuracy should not be considered in isolation.

## 5. Streamlit Application

The Streamlit application provides an interactive interface with four sections:

### Overview
Displays project information, dataset details, classification algorithms, and evaluation metrics.

### Prediction
Allows users to:
- Select one of the five trained models.
- Upload a customer dataset in CSV format.
- Generate subscription predictions.
- View and download prediction results.

### Evaluation
When the uploaded CSV contains the actual target column `y`, the application displays:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- MCC
- Confusion matrix
- Classification report

### Model Comparison
Displays the performance comparison table and chart for all five classification models.

The application loads saved model files. Uploaded data is used for prediction and evaluation; the application does not retrain models on uploaded test data.

## 6. Project Structure

```text
Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── bank.csv
├── test_data.csv
├── model_comparison.csv
├── ML_Assignment_2_2026AJ05122.ipynb
│
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```

## 7. Installation and Execution

### Prerequisites

- Python 3.10 or later
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 8. Project Files

- `app.py`: Streamlit application source code.
- `requirements.txt`: Python package dependencies.
- `ML_Assignment_2_2026AJ05122.ipynb`: Notebook containing data preparation, model training, and evaluation.
- `models/`: Saved trained classification models.
- `test_data.csv`: Test dataset used for prediction and evaluation.
- `model_comparison.csv`: Performance metrics for all five models.
- `bank.csv`: Original dataset used in the project.

## 9. Conclusion

This project demonstrates the implementation and comparison of five supervised machine learning classification algorithms on the Bank Marketing dataset.

The results show that model performance depends on the evaluation metric being considered. Although KNN achieved the highest accuracy, SVM provided the strongest balance among F1 Score, AUC, and MCC in this experiment.

The Streamlit application makes the trained models accessible through an interactive interface for customer subscription prediction and model evaluation.
