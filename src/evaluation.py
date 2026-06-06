# src/evaluation.py

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)
from sklearn.model_selection import cross_val_score
import numpy as np


def evaluate_classifier(model, X_test, y_test, average="weighted"):
    """
    Evaluate a classification model using common classification metrics.

    Parameters
    model : fitted classifier
        A trained classification model.

    X_test : array-like
        Test features.

    y_test : array-like
        True labels.

    average : str
        Averaging method for precision, recall, and F1.
        Common options: "binary", "macro", "weighted".
    """
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0)
    }

    return results


def print_classification_results(model, X_test, y_test, average="weighted"):
    """
    Print classification metrics, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)

    print("Classification Metrics")
    print("-" * 30)

    results = evaluate_classifier(model, X_test, y_test, average=average)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix")
    print("-" * 30)
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print("-" * 30)
    print(classification_report(y_test, y_pred, zero_division=0))


def evaluate_regressor(model, X_test, y_test):
    """
    Evaluate a regression model using common regression metrics.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2_score(y_test, y_pred)
    }

    return results


def print_regression_results(model, X_test, y_test):
    """
    Print common regression metrics.
    """
    results = evaluate_regressor(model, X_test, y_test)

    print("Regression Metrics")
    print("-" * 30)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")


def compare_models(models, X_test, y_test, task="classification", average="weighted"):
    """
    Compare multiple fitted models.

    Parameters
    models : dict
        Dictionary in the form:
        {
            "Model Name": fitted_model
        }

    task : str
        "classification" or "regression".
    """
    results = []

    for name, model in models.items():
        if task == "classification":
            metrics = evaluate_classifier(model, X_test, y_test, average=average)
        elif task == "regression":
            metrics = evaluate_regressor(model, X_test, y_test)
        else:
            raise ValueError("task must be either 'classification' or 'regression'.")

        metrics["model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)

    cols = ["model"] + [col for col in results_df.columns if col != "model"]
    results_df = results_df[cols]

    return results_df


def cross_validate_model(model, X, y, cv=5, scoring="accuracy"):
    """
    Run cross-validation and return mean and standard deviation.

    Useful for quick model stability checks.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    results = {
        "scores": scores,
        "mean_score": scores.mean(),
        "std_score": scores.std()
    }

    return results


def print_cross_validation_results(model, X, y, cv=5, scoring="accuracy"):
    """
    Print cross-validation results.
    """
    results = cross_validate_model(model, X, y, cv=cv, scoring=scoring)

    print(f"Cross-validation scoring: {scoring}")
    print("-" * 30)
    print("Scores:", results["scores"])
    print(f"Mean score: {results['mean_score']:.4f}")
    print(f"Standard deviation: {results['std_score']:.4f}")


def evaluate_clustering(X, labels):
    """
    Evaluate clustering using silhouette score.

    Note:
    Silhouette score only works when there are at least 2 clusters.
    """
    unique_labels = set(labels)

    if len(unique_labels) < 2:
        raise ValueError("Silhouette score requires at least 2 clusters.")

    score = silhouette_score(X, labels)

    return {
        "n_clusters": len(unique_labels),
        "silhouette_score": score
    }


def print_clustering_results(X, labels):
    """
    Print basic clustering evaluation.
    """
    results = evaluate_clustering(X, labels)

    print("Clustering Metrics")
    print("-" * 30)
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"Silhouette score: {results['silhouette_score']:.4f}")