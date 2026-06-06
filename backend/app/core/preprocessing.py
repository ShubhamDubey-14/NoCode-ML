"""DataFrame preprocessing and sklearn ColumnTransformer pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler


def _is_id_like(series: pd.Series) -> bool:
    """True if column looks like a unique identifier (nunique == len)."""
    return series.nunique(dropna=False) == len(series)


def _split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric, categorical


def _split_xy(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate features and target; drop ID-like feature columns."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    work = df.copy()
    y_raw = work[target_column]
    X = work.drop(columns=[target_column])

    id_cols = [c for c in X.columns if _is_id_like(X[c])]
    if id_cols:
        X = X.drop(columns=id_cols)

    if X.empty:
        raise ValueError("No feature columns remain after removing target and ID-like columns.")

    return X, y_raw, id_cols


def _coerce_numeric_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Try to parse object columns as numbers (e.g. iris sepal_length with NaNs as str)."""
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() >= max(1, int(0.5 * len(out))):
                out[col] = converted
    return out


def clean_dataframe(
    df: pd.DataFrame,
    target_column: str,
    imputation_strategy: str,
) -> pd.DataFrame:
    """
    Apply the same row/column cleaning as training, returning a human-readable CSV.

    Imputation fills missing values in-place (median / mode). No one-hot expansion.
    """
    if imputation_strategy not in ("mean_median", "drop"):
        raise ValueError("imputation_strategy must be 'mean_median' or 'drop'.")

    X, y_raw, _id_cols = _split_xy(df, target_column)
    X = _coerce_numeric_columns(X)

    if imputation_strategy == "drop":
        mask = ~(X.isna().any(axis=1) | y_raw.isna())
        X = X.loc[mask].reset_index(drop=True)
        y_raw = y_raw.loc[mask].reset_index(drop=True)
    else:
        numeric_cols, cat_cols = _split_feature_types(X)
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())
        for col in cat_cols:
            if X[col].isna().any():
                mode = X[col].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else ""
                X[col] = X[col].fillna(fill)
        if y_raw.isna().any():
            if pd.api.types.is_numeric_dtype(y_raw):
                y_raw = y_raw.fillna(y_raw.median())
            else:
                mode = y_raw.mode(dropna=True)
                y_raw = y_raw.fillna(mode.iloc[0] if len(mode) else "")

    cleaned = X.copy()
    cleaned[target_column] = y_raw.values
    return cleaned


def build_preprocessor(
    df: pd.DataFrame,
    target_column: str,
    imputation_strategy: str,
    scaling_method: str = "none",
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Separate X/y, drop ID-like columns, impute, scale, and build a ColumnTransformer.

    imputation_strategy: 'mean_median' | 'drop'
    scaling_method: 'none' | 'standardize' (Z-Score) | 'normalize' (MinMax [0,1])
    """
    X, y_raw, _id_cols = _split_xy(df, target_column)
    X = _coerce_numeric_columns(X)

    if imputation_strategy == "drop":
        mask = ~(X.isna().any(axis=1) | y_raw.isna())
        X = X.loc[mask].reset_index(drop=True)
        y_raw = y_raw.loc[mask].reset_index(drop=True)
    elif imputation_strategy != "mean_median":
        raise ValueError("imputation_strategy must be 'mean_median' or 'drop'.")

    numeric_cols, cat_cols = _split_feature_types(X)
    binary_cat: list[str] = []
    multi_cat: list[str] = []

    for col in cat_cols:
        if X[col].nunique(dropna=True) <= 2:
            binary_cat.append(col)
        else:
            multi_cat.append(col)

    transformers: list[tuple] = []

    if numeric_cols:
        num_steps: list[tuple] = []
        if imputation_strategy == "mean_median":
            num_steps.append(("imputer", SimpleImputer(strategy="median")))
        
        # Add scaling based on user selection
        if scaling_method == "standardize":
            num_steps.append(("scaler", StandardScaler()))
        elif scaling_method == "normalize":
            num_steps.append(("scaler", MinMaxScaler()))
        # If scaling_method == "none", no scaler is added
        
        transformers.append(
            ("num", Pipeline(num_steps) if num_steps else "passthrough", numeric_cols)
        )

    if binary_cat:
        bin_steps: list[tuple] = []
        if imputation_strategy == "mean_median":
            bin_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        # OrdinalEncoder ≈ per-column LabelEncoder, sklearn-pipeline safe
        bin_steps.append(
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        )
        transformers.append(("binary_cat", Pipeline(bin_steps), binary_cat))

    if multi_cat:
        multi_steps: list[tuple] = []
        if imputation_strategy == "mean_median":
            multi_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        multi_steps.append(
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        )
        transformers.append(("multi_cat", Pipeline(multi_steps), multi_cat))

    if not transformers:
        raise ValueError("Could not build any feature transformers for the dataset.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return X, y_raw, preprocessor


def encode_target(y: pd.Series) -> tuple[pd.Series, object | None]:
    """
    Map every class to consecutive integers 0..K-1 (required by XGBoost and stratified CV).
  """
    from sklearn.preprocessing import LabelEncoder

    n_classes = y.nunique(dropna=True)
    # Measurement columns mistaken as targets (e.g. sepal_length) have dozens of unique values.
    if n_classes > 15:
        raise ValueError(
            f"Target has {n_classes} distinct values — this looks like a numeric feature, "
            "not a class label. For Iris, choose **species** as the target column, not "
            "sepal_length / sepal_width / petal_length / petal_width."
        )

    le = LabelEncoder()
    encoded = le.fit_transform(y.astype(str))
    return pd.Series(encoded, index=y.index, name=y.name), le
