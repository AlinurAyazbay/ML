"""
Train Iris classification model with MLflow experiment tracking
and Model Registry integration.

This script:
  1. Trains a Logistic Regression model on the Iris dataset
  2. Logs hyperparameters, metrics, and the model artifact to MLflow
  3. Registers the model in the MLflow Model Registry

Usage:
    python train.py
    python train.py --C 0.5 --max-iter 300 --solver liblinear
"""

import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "iris-classification"
REGISTERED_MODEL_NAME = "iris-classifier"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Iris classifier with MLflow")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max iterations for the solver")
    parser.add_argument("--solver", type=str, default="lbfgs",
                        choices=["lbfgs", "newton-cg", "saga", "sag"],
                        help="Optimization algorithm (multiclass-compatible)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def train_model(args):
    # -----------------------------------------------------------------------
    # Set up MLflow
    # -----------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    # -----------------------------------------------------------------------
    # Load & split data
    # -----------------------------------------------------------------------
    print("\nLoading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # -----------------------------------------------------------------------
    # Train inside an MLflow run
    # -----------------------------------------------------------------------
    with mlflow.start_run() as run:
        print(f"\nMLflow run_id: {run.info.run_id}")

        # Log hyperparameters
        params = {
            "C": args.C,
            "max_iter": args.max_iter,
            "solver": args.solver,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "model_type": "LogisticRegression",
            "dataset": "iris",
            "n_features": X.shape[1],
            "n_classes": len(iris.target_names),
        }
        mlflow.log_params(params)
        print("Logged parameters:", params)

        # Train model
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            solver=args.solver,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
        }
        mlflow.log_metrics(metrics)

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

        # -------------------------------------------------------------------
        # Log the sklearn model AND register it in the Model Registry
        # -------------------------------------------------------------------
        print(f"Registering model as '{REGISTERED_MODEL_NAME}' in Model Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_test[:1],
        )

        # -------------------------------------------------------------------
        # Also save a local model.joblib so the FastAPI service can load it
        # without hitting the registry at startup (simple + reliable).
        # -------------------------------------------------------------------
        local_model_path = os.getenv("LOCAL_MODEL_PATH", "model.joblib")
        joblib.dump(model, local_model_path)
        mlflow.log_artifact(local_model_path, artifact_path="joblib_artifact")
        print(f"Saved local model artifact to: {local_model_path}")

        print(f"\nDone. View the run in the MLflow UI: {MLFLOW_TRACKING_URI}")
        return model, metrics


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
