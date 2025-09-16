import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
import os, json

def train_model(data_path, out_dir):
    # Load dataset
    df = pd.read_csv(data_path)
    print("Dataset loaded:", df.shape)

    X = df.drop("label", axis=1)
    y = df["label"]

    # Encode target labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Model
    model = LGBMClassifier(random_state=42)

    # Hyperparameter tuning
    param_dist = {
        "num_leaves": [20, 31, 50],
        "max_depth": [-1, 5, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200]
    }

    search = RandomizedSearchCV(
        model, param_distributions=param_dist,
        n_iter=10, scoring="accuracy", cv=3, random_state=42, n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best params:", search.best_params_)

    # Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))

    meta = {
        "accuracy": acc,
        "best_params": search.best_params_,
        "n_classes": len(le.classes_)
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\nModel & artifacts saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    train_model(args.data_path, args.out_dir)
