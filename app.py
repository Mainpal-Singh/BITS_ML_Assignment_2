
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide"
)

# ==================================================
# 2. CONSTANTS
# ==================================================

MODEL_DIR = "models"
COMPARISON_FILE = "model_comparison.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

REQUIRED_FEATURES = [
    "age", "job", "marital", "education",
    "default", "balance", "housing", "loan",
    "contact", "day", "month", "campaign",
    "pdays", "previous", "poutcome"
]

METRICS = [
    "Accuracy", "Precision", "Recall",
    "F1 Score", "AUC", "MCC"
]

# ==================================================
# 3. LOAD TRAINED MODELS
# ==================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        file_path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found: {file_path}"
            )

        loaded_models[model_name] = joblib.load(file_path)

    return loaded_models


try:
    models = load_models()

except Exception as e:
    st.error(f"Unable to load trained models: {e}")
    st.stop()


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def convert_labels_to_binary(labels):

    """Convert yes/no or 0/1 labels into binary values."""

    converted = []

    for value in labels:

        value = str(value).strip().lower()

        if value in ["1", "yes"]:
            converted.append(1)

        elif value in ["0", "no"]:
            converted.append(0)

        else:
            raise ValueError(
                f"Unexpected class label: {value}"
            )

    return np.array(converted)


def get_prediction_scores(model, X):

    """Get continuous scores for ROC-AUC calculation."""

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        classes = list(model.classes_)

        if 1 in classes:
            positive_index = classes.index(1)
            return probabilities[:, positive_index]

        if "yes" in classes:
            positive_index = classes.index("yes")
            return probabilities[:, positive_index]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    raise ValueError(
        "This model does not provide probability "
        "or decision scores for AUC calculation."
    )


def calculate_evaluation(model, X, actual, predictions):

    """Calculate all six classification metrics."""

    accuracy = accuracy_score(actual, predictions)

    precision = precision_score(
        actual, predictions, zero_division=0
    )

    recall = recall_score(
        actual, predictions, zero_division=0
    )

    f1 = f1_score(
        actual, predictions, zero_division=0
    )

    mcc = matthews_corrcoef(actual, predictions)

    auc = None

    if len(np.unique(actual)) == 2:

        scores = get_prediction_scores(model, X)
        auc = roc_auc_score(actual, scores)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc,
        "MCC": mcc
    }


# ==================================================
# 5. APPLICATION HEADER
# ==================================================

st.title("🏦 Bank Marketing Classification")

st.write(
    "A Machine Learning application to predict whether "
    "a customer will subscribe to a bank term deposit."
)

st.caption(
    "Machine Learning Assignment 2 | "
    "Five Classification Algorithms"
)

st.divider()

# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🏦 Bank Marketing")

    st.success("All five trained models loaded")

    st.divider()

    st.subheader("Available Models")

    for model_name in models:
        st.write(f"✓ {model_name}")

    st.divider()

    st.caption(
        "Models are loaded from saved files. "
        "Uploaded data is used only for prediction "
        "and evaluation, not for retraining."
    )


# ==================================================
# 7. FOUR DASHBOARD TABS
# ==================================================

overview_tab, prediction_tab, evaluation_tab, comparison_tab = (
    st.tabs([
        "🏠 Overview",
        "🔮 Prediction",
        "📊 Evaluation",
        "📈 Model Comparison"
    ])
)


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with overview_tab:

    st.header("Project Overview")

    st.write(
        """
        This application uses five supervised machine learning
        classification algorithms to predict whether a bank
        customer will subscribe to a term deposit.
        """
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", "4,521")
    col2.metric("Input Features", "15")
    col3.metric("Target Classes", "2")

    st.write("**Target variable:** `y`")

    st.write(
        "- `yes`: Customer subscribed to a term deposit."
    )

    st.write(
        "- `no`: Customer did not subscribe to a term deposit."
    )

    st.divider()

    st.subheader("Classification Algorithms")

    model_columns = st.columns(5)

    for column, model_name in zip(
        model_columns, models.keys()
    ):

        with column:
            st.info(model_name)

    st.divider()

    st.subheader("Evaluation Metrics")

    st.write(
        """
        The models are evaluated using six classification
        performance measures:
        """
    )

    st.markdown(
        """
        - **Accuracy:** Overall prediction correctness.
        - **Precision:** Reliability of positive predictions.
        - **Recall:** Ability to identify actual subscribers.
        - **F1 Score:** Harmonic mean of precision and recall.
        - **AUC:** Ability to distinguish between the two classes.
        - **MCC:** Balanced measure of binary classification quality.
        """
    )


# ==================================================
# TAB 2: PREDICTION
# ==================================================

with prediction_tab:

    st.header("Customer Subscription Prediction")

    selected_model_name = st.selectbox(
        "Select Classification Model",
        list(models.keys())
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV Dataset",
        type=["csv"],
        help="Upload a CSV containing the 15 required input features."
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )

            st.success("CSV uploaded successfully!")

            st.subheader("Dataset Preview")

            st.dataframe(
                uploaded_data.head(),
                use_container_width=True
            )

            st.write(
                f"Dataset shape: {uploaded_data.shape}"
            )

            missing_features = [
                col for col in REQUIRED_FEATURES
                if col not in uploaded_data.columns
            ]

            if missing_features:

                st.error(
                    "Missing required features: "
                    + ", ".join(missing_features)
                )

            elif uploaded_data.empty:

                st.error("The uploaded dataset is empty.")

            else:

                st.success(
                    "All 15 required input features are available."
                )

                X_new = uploaded_data[
                    REQUIRED_FEATURES
                ].copy()

                if st.button(
                    "Generate Predictions",
                    type="primary"
                ):

                    selected_model = models[
                        selected_model_name
                    ]

                    # Generate predictions using saved model
                    raw_predictions = selected_model.predict(
                        X_new
                    )

                    predictions = convert_labels_to_binary(
                        raw_predictions
                    )

                    prediction_labels = np.where(
                        predictions == 1, "yes", "no"
                    )

                    results = uploaded_data.copy()

                    results[
                        "Predicted Subscription"
                    ] = prediction_labels

                    # Store prediction results
                    st.session_state["results"] = results

                    st.session_state["predictions"] = predictions

                    st.session_state["selected_model"] = (
                        selected_model_name
                    )

                    st.session_state["uploaded_data"] = (
                        uploaded_data.copy()
                    )

                    # Calculate evaluation if actual target exists
                    if "y" in uploaded_data.columns:

                        actual = convert_labels_to_binary(
                            uploaded_data["y"]
                        )

                        evaluation = calculate_evaluation(
                            selected_model,
                            X_new,
                            actual,
                            predictions
                        )

                        st.session_state["actual"] = actual
                        st.session_state["evaluation"] = evaluation

                    else:

                        st.session_state.pop(
                            "evaluation", None
                        )

                        st.session_state.pop(
                            "actual", None
                        )

                    st.success(
                        "Predictions generated successfully!"
                    )

            # Display stored results
            if (
                "results" in st.session_state
                and st.session_state.get("selected_model")
                == selected_model_name
            ):

                results = st.session_state["results"]

                st.subheader("Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                predictions = st.session_state["predictions"]

                predicted_yes = int(
                    np.sum(predictions == 1)
                )

                predicted_no = int(
                    np.sum(predictions == 0)
                )

                st.subheader("Prediction Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Customers",
                    len(predictions)
                )

                col2.metric(
                    "Predicted Yes",
                    predicted_yes
                )

                col3.metric(
                    "Predicted No",
                    predicted_no
                )

                st.download_button(
                    "Download Prediction Results",
                    data=results.to_csv(index=False),
                    file_name="bank_predictions.csv",
                    mime="text/csv"
                )

                if "y" not in results.columns:

                    st.info(
                        "Actual target column 'y' is not present. "
                        "Predictions are available, but evaluation "
                        "metrics cannot be calculated."
                    )

        except Exception as e:

            st.error(
                f"Unable to process uploaded dataset: {e}"
            )


# ==================================================
# TAB 3: EVALUATION
# ==================================================

with evaluation_tab:

    st.header("Model Evaluation")

    if "evaluation" not in st.session_state:

        st.info(
            "Generate predictions using a CSV with an actual "
            "target column 'y' in the Prediction tab first."
        )

    else:

        evaluation = st.session_state["evaluation"]

        selected_model_name = st.session_state[
            "selected_model"
        ]

        st.subheader(
            f"Evaluation Results: {selected_model_name}"
        )

        metric_columns = st.columns(3)

        for i, metric in enumerate(METRICS):

            value = evaluation[metric]

            with metric_columns[i % 3]:

                st.metric(
                    metric,
                    "N/A" if value is None else f"{value:.4f}"
                )

        st.divider()

        actual = st.session_state["actual"]
        predictions = st.session_state["predictions"]

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            actual,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"{selected_model_name} - Confusion Matrix"
        )

        st.pyplot(fig)
        plt.close(fig)

        # Classification Report
        st.subheader("Classification Report")

        report = classification_report(
            actual,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            zero_division=0,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )


# ==================================================
# TAB 4: MODEL COMPARISON
# ==================================================

with comparison_tab:

    st.header("Model Performance Comparison")

    if os.path.exists(COMPARISON_FILE):

        comparison_df = pd.read_csv(
            COMPARISON_FILE
        )

        required_comparison_columns = [
            "Model"
        ] + METRICS

        missing_columns = [
            col for col in required_comparison_columns
            if col not in comparison_df.columns
        ]

        if missing_columns:

            st.error(
                "Comparison file is missing columns: "
                + ", ".join(missing_columns)
            )

        else:

            
            st.subheader("Performance Comparison Table")

            # Format all evaluation metrics to four decimal places
            formatted_comparison_df = comparison_df.style.format(
            {
                metric: "{:.4f}"
                for metric in METRICS
            }
        )

            st.dataframe(
                formatted_comparison_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Model Comparison CSV",
                data=comparison_df.to_csv(index=False),
                file_name="model_comparison.csv",
                mime="text/csv"
            )

            st.divider()

            st.subheader("Performance Comparison Chart")

            chart_df = comparison_df.set_index(
                "Model"
            )[METRICS]

            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(chart_df))
            bar_width = 0.12

            for i, metric in enumerate(METRICS):

                positions = (
                    x + (i - 2.5) * bar_width
                )

                ax.bar(
                    positions,
                    chart_df[metric],
                    bar_width,
                    label=metric
                )

            ax.set_title(
                "Performance Comparison of Five Classification Models"
            )

            ax.set_xlabel("Classification Model")
            ax.set_ylabel("Score")

            ax.set_xticks(x)
            ax.set_xticklabels(
                chart_df.index,
                rotation=15
            )

            ax.set_ylim(0, 1.0)

            ax.legend(
                title="Evaluation Metrics",
                bbox_to_anchor=(1.02, 1),
                loc="upper left"
            )

            ax.grid(
                axis="y",
                alpha=0.3
            )

            ax.set_axisbelow(True)

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

    else:

        st.warning(
            "model_comparison.csv was not found."
        )


# ==================================================
# 8. FOOTER
# ==================================================

st.divider()

st.caption(
    "Bank Marketing Classification | "
    "Machine Learning Assignment 2"
)
