# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def basic_info(df):
    """
    Display basic dataset information.

    Useful for quick notebook EDA.
    """
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())


def check_missing_values(df):
    """
    Return a dataframe showing missing value counts and percentages.
    """
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    })

    return missing_df.sort_values(by="missing_count", ascending=False)


def split_features_target(df, target_column):
    """
    Split a dataframe into features X and target y.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def make_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Create a train/test split.

    For classification, pass stratify=y if class balance matters.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )


def scale_data(X_train, X_test, method="standard"):
    """
    Scale train and test data.

    Parameters
    method : str
        "standard" for StandardScaler or "minmax" for MinMaxScaler.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be either 'standard' or 'minmax'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def build_preprocessing_pipeline(numeric_features, categorical_features=None, scale_numeric=True):
    """
    Build a simple preprocessing pipeline for tabular ML.

    Handles:
    - numeric missing values with median imputation
    - optional numeric scaling
    - categorical missing values with most frequent value
    - categorical one-hot encoding

    Parameters
    numeric_features : list
        Numeric column names.

    categorical_features : list, optional
        Categorical column names.

    scale_numeric : bool
        Whether to apply StandardScaler to numeric columns.
    """
    if categorical_features is None:
        categorical_features = []

    if scale_numeric:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def get_numeric_categorical_columns(df, target_column=None):
    """
    Automatically separate numeric and categorical columns.

    Useful for quick experiments.
    """
    if target_column is not None:
        df = df.drop(columns=[target_column])

    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return numeric_features, categorical_features