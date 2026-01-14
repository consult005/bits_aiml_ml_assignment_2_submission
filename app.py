import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Streamlit page config
# -------------------------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 – Bank Marketing Classification")

# -------------------------------------------------
# Sample CSV download
# -------------------------------------------------
SAMPLE_PATH = "sample/sample_test.csv"

st.subheader("Sample Test CSV")
st.write("If you don't have a test file, download a sample and upload it back to evaluate the models.")

if os.path.exists(SAMPLE_PATH):
    with open(SAMPLE_PATH, "rb") as f:
        st.download_button(
            label="⬇️ Download Sample Test CSV",
            data=f,
            file_name="sample_test.csv",
            mime="text/csv"
        )
else:
    st.warning("Sample file not found in repo: sample/sample_test.csv")

st.write(
    """
This Streamlit application demonstrates multiple classification models trained on the Bank Marketing dataset.
Upload test data, select a model, and view evaluation results.
"""
)

# -------------------------------------------------
# Model registry
# -------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "model/artifacts/logistic_regression.pkl",
    "Decision Tree": "model/artifacts/decision_tree.pkl",
    "KNN": "model/artifacts/knn.pkl",
    "Naive Bayes": "model/artifacts/naive_bayes.pkl",
    "Random Forest": "model/artifacts/random_forest.pkl",
    "XGBoost": "model/artifacts/xgboost.pkl",
}

# -------------------------------------------------
# Caching helpers (critical for Streamlit resource usage)
# -------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def read_csv_auto_sep(uploaded_file) -> pd.DataFrame:
    """
    Try reading CSV using common separators.
    - bank-additional-full is ';' separated
    - generated test.csv is usually ',' separated
    """
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, sep=";")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)  # default comma

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (must contain target column 'y')", type=["csv"])

run_eval = st.sidebar.button("Run Evaluation")

# -------------------------------------------------
# Main flow
# -------------------------------------------------
if uploaded_file is None:
    st.info("Please upload the test CSV file. Tip: Use the `data/test.csv` generated during model training.")
    st.stop()

# Read CSV (cached)
df = read_csv_auto_sep(uploaded_file)

# Resource protection: row/col limits (stop early)
MAX_ROWS = 2000
MAX_COLS = 200

if len(df) > MAX_ROWS:
    st.error(f"Please upload test data only (max {MAX_ROWS} rows). Your file has {len(df)} rows.")
    st.stop()

if df.shape[1] > MAX_COLS:
    st.error(f"Too many columns ({df.shape[1]}). Please upload the correct test file with expected columns.")
    st.stop()

# Ensure target column exists
if "y" not in df.columns:
    st.error(f"Uploaded CSV must include the target column 'y'. Found columns: {', '.join(df.columns)}")
    st.stop()

# Convert y to numeric 0/1 if needed
y_true_raw = df["y"]
if y_true_raw.dtype == object:
    y_true = y_true_raw.map({"yes": 1, "no": 0})
else:
    y_true = y_true_raw

if y_true.isna().any():
    st.error("Target column 'y' has unexpected values. Allowed: yes/no or 0/1.")
    st.stop()

X = df.drop(columns=["y"])

# Don’t run heavy work on every widget change
if not run_eval:
    st.warning("Select model + upload file, then click **Run Evaluation**.")
    st.stop()

# Load model (cached)
model_path = MODEL_FILES[selected_model]
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Ensure artifacts are committed to GitHub.")
    st.stop()

model = load_model(model_path)

# Predict
y_pred = model.predict(X)

# Probabilities for AUC
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X)[:, 1]
else:
    scores = model.decision_function(X)
    y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

metrics = compute_metrics(y_true, y_pred, y_proba)

# -------------------------------------------------
# Display results (better UI)
# -------------------------------------------------
st.subheader(f"Results for: {selected_model}")

# KPI cards
c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
c2.metric("AUC", f"{metrics['AUC']:.4f}")
c3.metric("Precision", f"{metrics['Precision']:.4f}")
c4.metric("Recall", f"{metrics['Recall']:.4f}")
c5.metric("F1 Score", f"{metrics['F1']:.4f}")
c6.metric("MCC", f"{metrics['MCC']:.4f}")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (no)", "1 (yes)"])
    ax.set_yticklabels(["0 (no)", "1 (yes)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)  # important: prevents memory growth on reruns

with right:
    st.subheader("Classification Report")
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)
