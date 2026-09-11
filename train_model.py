"""
train_model.py
---------------
Trains a Machine Learning model to predict house prices and saves the
complete trained pipeline (preprocessing + model) to disk using Joblib.

Steps performed:
    1. Load the dataset (data/house_prices.csv)
    2. Separate features (X) and target (y)
    3. Build a preprocessing pipeline (scaling + one-hot encoding)
    4. Split into train/test sets
    5. Train two candidate models: Linear Regression & Random Forest
    6. Evaluate both using MAE, RMSE, R²
    7. Pick the better-performing model
    8. Save the full pipeline (preprocessing + model) to model/house_price_model.pkl

Run:
    python train_model.py

Requirements:
    Run generate_dataset.py first if data/house_prices.csv does not exist.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Resolve paths relative to this file so the script works regardless
# of the current working directory (important for cross-platform use).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "house_prices.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")

RANDOM_SEED = 42

NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "parking", "house_age"]
CATEGORICAL_FEATURES = ["location"]
TARGET_COLUMN = "price"


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from CSV, with a clear error if it's missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please run 'python generate_dataset.py' first to create it."
        )
    df = pd.read_csv(path)

    # Basic data cleaning: drop any rows with missing values
    # (the synthetic dataset shouldn't have any, but this makes the
    # script robust if a real-world dataset is swapped in later).
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"⚠️  Dropped {before - after} rows containing missing values.")

    return df


def build_pipeline(model) -> Pipeline:
    """
    Build a full preprocessing + model pipeline.

    - Numeric features are scaled with StandardScaler.
    - Categorical features (location) are one-hot encoded.

    Wrapping everything in a single sklearn Pipeline means the exact
    same preprocessing used during training is automatically applied
    during prediction in the Streamlit app - no manual re-implementation
    needed, and no risk of train/test preprocessing mismatch.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate(y_true, y_pred, model_name: str) -> dict:
    """Compute and print MAE, RMSE and R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Evaluation:")
    print(f"   MAE  (Mean Absolute Error) : {mae:.2f} Lakhs")
    print(f"   RMSE (Root Mean Sq. Error) : {rmse:.2f} Lakhs")
    print(f"   R² Score                   : {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    print("🔄 Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"   Loaded {len(df)} rows.")

    # Separate features and target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    # Train/test split - done BEFORE any preprocessing is fit, so there
    # is no data leakage from the test set into the training process.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"   Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # --- Candidate 1: Linear Regression (simple, interpretable baseline) ---
    lr_pipeline = build_pipeline(LinearRegression())
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_scores = evaluate(y_test, lr_preds, "Linear Regression")

    # --- Candidate 2: Random Forest Regressor (captures non-linear patterns) ---
    # max_depth is capped so the saved model file stays small (a few MB
    # instead of 40+ MB) while barely affecting accuracy - this keeps
    # the repository lightweight and GitHub/Streamlit Cloud friendly.
    rf_pipeline = build_pipeline(
        RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_SEED
        )
    )
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_scores = evaluate(y_test, rf_preds, "Random Forest Regressor")

    # --- Pick the better model based on R² score on the test set ---
    if rf_scores["r2"] >= lr_scores["r2"]:
        best_pipeline = rf_pipeline
        best_name = "Random Forest Regressor"
        best_scores = rf_scores
    else:
        best_pipeline = lr_pipeline
        best_name = "Linear Regression"
        best_scores = lr_scores

    print(f"\n🏆 Best model selected: {best_name} (R² = {best_scores['r2']:.4f})")

    # Save the winning pipeline (preprocessing + model together)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "metrics": best_scores,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "locations": sorted(df["location"].unique().tolist()),
        },
        MODEL_PATH,
    )
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print("   The Streamlit app will automatically use this trained model.")


if __name__ == "__main__":
    main()
