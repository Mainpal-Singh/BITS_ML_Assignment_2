# Bank Marketing Classification – Machine Learning Assignment 2

## a. Problem Statement

Build and evaluate five supervised machine learning classification models that predict whether a bank customer will subscribe to a term deposit. The project uses one public classification dataset, compares the models using six evaluation metrics, and demonstrates the trained models through an interactive Streamlit application. The application loads saved models and evaluates uploaded test data without retraining on that data.

## b. Dataset Description

- **Dataset:** Bank Marketing Dataset
- **Source:** UCI Machine Learning Repository
- **File used:** `bank.csv`
- **Number of records:** 4,521
- **Input features used:** 15
- **Target variable:** `y`
- **Target classes:** `yes` (subscribed) and `no` (not subscribed)
- **CSV delimiter:** Semicolon (`;`)

The dataset contains customer demographic information, account details, and information about previous marketing campaigns. The original `duration` feature was excluded from model inputs because it is only known after a marketing contact takes place.

## c. GitHub Repository Link

[https://github.com/Mainpal-Singh/BITS_ML_Assignment_2](https://github.com/Mainpal-Singh/BITS_ML_Assignment_2)

The repository contains the application source code, notebook, dependency file, dataset, test data, model comparison results, and saved model pipelines for all five classifiers.

## d. Models Used and Performance Comparison

The following five classification models were implemented on the same dataset and evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7050 | 0.7350 | 0.2237 | 0.6346 | 0.3308 | 0.2373 |
| Decision Tree | 0.7724 | 0.6788 | 0.2287 | 0.4135 | 0.2945 | 0.1827 |
| K-Nearest Neighbors (KNN) | 0.8762 | 0.6112 | 0.2778 | 0.0481 | 0.0820 | 0.0727 |
| Naive Bayes | 0.8099 | 0.7166 | 0.2763 | 0.4038 | 0.3281 | 0.2274 |
| Support Vector Machine (SVM) | 0.7901 | 0.7502 | 0.2913 | 0.5769 | 0.3871 | 0.3002 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the highest recall (0.6346), identifying the largest proportion of actual positive subscriptions among the five models. Its F1 score was 0.3308 and AUC was 0.7350. |
| Decision Tree | Achieved an accuracy of 0.7724 and recall of 0.4135. Its F1 score (0.2945) and AUC (0.6788) were lower than those of Logistic Regression and SVM. |
| KNN | Achieved the highest accuracy (0.8762), but its recall (0.0481) and F1 score (0.0820) were very low. This indicates that accuracy alone is not sufficient to assess performance on this imbalanced dataset. |
| Naive Bayes | Achieved an accuracy of 0.8099, recall of 0.4038, F1 score of 0.3281, and AUC of 0.7166, showing moderate performance across the metrics. |
| Support Vector Machine | Achieved the highest F1 score (0.3871), AUC (0.7502), and MCC (0.3002). It provides the strongest overall balance among these models when considering these metrics together. |

### Overall winner for this dataset

**Support Vector Machine (SVM)** is the overall winner based on its highest F1 Score, AUC, and MCC. Logistic Regression is strongest in recall, while KNN has the highest accuracy. Since the target classes are imbalanced, the selection is based on multiple metrics rather than accuracy alone.

## Streamlit Application

The deployed application provides:
- CSV upload for test data.
- A dropdown to select one of the five saved classification models.
- Prediction output and downloadable predictions.
- Evaluation metrics when the uploaded CSV contains the actual target column `y`.
- Confusion matrix and classification report.
- A comparison table and chart for all five models.

The app loads saved trained pipelines and does not retrain models on uploaded test data.

**Live Streamlit application:** [https://bitsmlassignment2-qtrwvenvftfpmnd3k8ukcw.streamlit.app/](https://bitsmlassignment2-qtrwvenvftfpmnd3k8ukcw.streamlit.app/)

## Project Structure

```text
Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── bank.csv
├── test_data.csv
├── model_comparison.csv
├── ML_Assignment_2_2026AJ05122.ipynb
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```

## Installation and Execution

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Conclusion

This project implements and compares five supervised classification algorithms on the Bank Marketing dataset. The results demonstrate that the best model depends on the evaluation objective. SVM achieved the strongest combined performance in F1 Score, AUC, and MCC, while Logistic Regression achieved the highest recall and KNN achieved the highest accuracy.
