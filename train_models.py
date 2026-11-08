"""Train 5 classifiers on the UCI Student Dropout dataset and save them to model/.

Run once:  python train_models.py
Outputs:   model/*.joblib, model/metrics.csv, test_data.csv
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
TARGET = "Target"
SEED = 7

# Scaling lives inside each pipeline, so the app feeds raw CSV columns straight in.
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
}


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=model.classes_),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted"),
        "F1": f1_score(y, pred, average="weighted"),
        "MCC": matthews_corrcoef(y, pred),
    }


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "raw" / "data.csv", sep=";")
    df.columns = [c.strip() for c in df.columns]  # one header has a stray tab
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    (ROOT / "model").mkdir(exist_ok=True)
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, ROOT / "model" / f"{name.lower().replace(' ', '_')}.joblib")
        rows.append({"ML Model Name": name, **evaluate(model, X_test, y_test)})

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(ROOT / "model" / "metrics.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(ROOT / "test_data.csv", index=False)
    print(metrics.to_string(index=False))
