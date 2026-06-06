# src/plotting.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_2d_data(X, labels=None, title="2D Data", xlabel="Feature 1", ylabel="Feature 2"):
    """
    Plot a simple 2D scatter plot.

    Parameters
    X : array-like of shape (n_samples, 2)
        2D input data.

    labels : array-like, optional
        Labels or cluster assignments used for colouring points.

    title : str
        Plot title.
    """
    plt.figure(figsize=(7, 5))

    if labels is None:
        plt.scatter(X[:, 0], X[:, 1], s=30, alpha=0.8)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, alpha=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.show()


def plot_clusters(X, labels, title="Cluster Assignments"):
    """
    Plot clustering results for 2D data.

    Parameters
    X : array-like of shape (n_samples, 2)
        2D input data.

    labels : array-like
        Cluster labels.
    """
    plot_2d_data(
        X=X,
        labels=labels,
        title=title,
        xlabel="Feature 1",
        ylabel="Feature 2"
    )


def plot_cluster_centers(X, labels, centers, title="Clusters with Centers"):
    """
    Plot 2D clusters and their centers.

    Useful for K-Means experiments.
    """
    plt.figure(figsize=(7, 5))

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, alpha=0.75)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        edgecolor="black",
        linewidth=1.2,
        label="Centers"
    )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, title="Confusion Matrix"):
    """
    Plot a confusion matrix for a fitted classifier.
    """
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.grid(False)
    plt.show()


def plot_feature_importance(feature_names, importances, title="Feature Importance", top_n=None):
    """
    Plot feature importances from models such as Decision Tree or Random Forest.

    Parameters
    feature_names : list
        Names of the input features.

    importances : array-like
        Importance score for each feature.

    top_n : int, optional
        If provided, only the top_n features are shown.
    """
    feature_names = np.array(feature_names)
    importances = np.array(importances)

    sorted_idx = np.argsort(importances)

    if top_n is not None:
        sorted_idx = sorted_idx[-top_n:]

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names[sorted_idx], importances[sorted_idx])
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.3)
    plt.show()


def plot_train_test_scores(train_scores, test_scores, x_values, xlabel="Model Complexity", title="Train vs Test Score"):
    """
    Plot training and testing scores against some changing value.

    Useful for showing overfitting/underfitting.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(x_values, train_scores, marker="o", label="Train Score")
    plt.plot(x_values, test_scores, marker="o", label="Test Score")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()