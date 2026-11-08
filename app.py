"""Streamlit app: upload test CSV -> pick a saved model -> see metrics + confusion matrix."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import TARGET, evaluate

MODEL_DIR = Path(__file__).parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Dropout & Academic Success — Model Lab")
st.caption("UCI dataset #697 · 3 classes: Dropout / Enrolled / Graduate · models are pre-trained, nothing is retrained here")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


with st.sidebar:
    st.header("1 · Upload test data")
    upload = st.file_uploader("CSV with all 36 feature columns + 'Target'", type="csv")
    st.header("2 · Choose a model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES))
    st.download_button("Download sample test_data.csv",
                       (Path(__file__).parent / "test_data.csv").read_bytes(),
                       "test_data.csv", "text/csv")

tab_eval, tab_compare = st.tabs(["Evaluate on uploaded data", "Comparison of all 5 models"])

with tab_compare:
    st.subheader("Hold-out test metrics (20% stratified split)")
    st.dataframe(pd.read_csv(MODEL_DIR / "metrics.csv").set_index("ML Model Name")
                 .style.highlight_max(axis=0, color="#c8e6c9"), width="stretch")

with tab_eval:
    if upload is None:
        st.info("Upload a test CSV from the sidebar (or download the sample one first).")
        st.stop()

    data = pd.read_csv(upload)
    data.columns = [c.strip() for c in data.columns]
    model = load_model(model_name)
    expected = list(model.feature_names_in_)
    missing = [c for c in expected + [TARGET] if c not in data.columns]
    if missing:
        st.error(f"Uploaded CSV is missing columns: {missing}")
        st.stop()

    X, y = data[expected], data[TARGET]
    st.write(f"**{model_name}** on {len(data)} uploaded rows")

    m = evaluate(model, X, y)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    pred = model.predict(X)
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, pred, labels=model.classes_)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    with st.expander("Row-level predictions"):
        st.dataframe(data.assign(Predicted=pred)[[TARGET, "Predicted"] + expected[:5]])
