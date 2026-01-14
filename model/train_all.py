import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

from datapreperation.config import RANDOM_STATE, TEST_SIZE, ARTIFACT_DIR, METRICS_CSV, TEST_OUT_CSV, TARGET_COL

from datapreperation.data import load_dataset
from datapreperation.preprocess import build_preprocessor
from datapreperation.metrics import compute_binary_metrics

from model.logistic_regression import build_model as build_lr
from model.decision_tree import build_model as build_dt
from model.knn import build_model as build_knn
from model.naive_bayes import build_model as build_nb
from model.random_forest import build_model as build_rf
from model.xgboost_model import build_model as build_xgb


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    X, y = load_dataset()
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": build_lr(),
        "Decision Tree": build_dt(),
        "KNN": build_knn(),
        "Naive Bayes": build_nb(),
        "Random Forest": build_rf(),
        "XGBoost": build_xgb(),
    }

    rows = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

        row = {"Model": name}
        row.update(compute_binary_metrics(y_test, y_pred, y_proba))
        rows.append(row)

        # Save model artifact for Streamlit
        joblib.dump(pipe, f"{ARTIFACT_DIR}/{safe_name(name)}.pkl")

        # Optional console outputs
        print(f"\n=== {name} ===")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save metrics table for README
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    # Save test.csv for Streamlit upload (include y so metrics can be computed)
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(TEST_OUT_CSV, index=False)

    print("\n Saved metrics:", METRICS_CSV)
    print("Saved models in:", ARTIFACT_DIR)
    print("Saved test set:", TEST_OUT_CSV)


if __name__ == "__main__":
    main()
