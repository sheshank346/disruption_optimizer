"""
model.py
Trains an XGBoost classifier that predicts a passenger's rebooking priority
score (0-1 probability) from booking/flight features. This score feeds into
the OR-Tools optimizer as passenger "weights" -- higher predicted priority
means the optimizer tries harder to give them a good rebooking slot.

We train on a large simulated dataset spanning many disrupted flights so the
model learns general patterns rather than memorizing one flight.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from passengers import generate_passengers

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "priority_model.joblib")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "encoders.joblib")

FEATURE_COLS = [
    "fare_class_enc", "loyalty_tier_enc", "booking_lead_days",
    "has_checked_bag", "group_size", "past_no_show_rate",
    "has_connecting_flight", "connection_buffer_min",
]


def build_training_dataset(num_flights=150, seed=42) -> pd.DataFrame:
    """
    Simulates passenger manifests across many disrupted flights to build a
    training set large enough for XGBoost to learn meaningful patterns.
    """
    rng = np.random.default_rng(seed)
    all_passengers = []
    for i in range(num_flights):
        num_pax = rng.integers(80, 220)
        conn_available = rng.random() > 0.15  # most flights have some onward connections
        df = generate_passengers(
            flight_id=f"SIM{i:04d}",
            num_passengers=num_pax,
            connecting_flight_available=conn_available,
            seed=int(rng.integers(0, 1_000_000)),
        )
        all_passengers.append(df)

    return pd.concat(all_passengers, ignore_index=True)


def prepare_features(df: pd.DataFrame, encoders=None, fit=False):
    """Encodes categorical columns and returns (X, encoders)."""
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in ["fare_class", "loyalty_tier"]:
        if fit:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[f"{col}_enc"] = le.transform(df[col])

    df["has_checked_bag"] = df["has_checked_bag"].astype(int)
    df["has_connecting_flight"] = df["has_connecting_flight"].astype(int)

    X = df[FEATURE_COLS]
    return X, encoders


def train_model():
    print("Building simulated training dataset across many disrupted flights...")
    data = build_training_dataset(num_flights=150)
    print(f"Total training passengers: {len(data)}")

    X, encoders = prepare_features(data, fit=True)
    y = data["rebooking_priority"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = (preds_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, preds_proba)
    print(f"\nTest ROC-AUC: {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds))

    print("\nFeature importances:")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importances)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model, encoders


def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders


def predict_priority(passengers_df: pd.DataFrame, model, encoders) -> pd.Series:
    """Returns predicted priority probability (0-1) for each passenger."""
    X, _ = prepare_features(passengers_df, encoders=encoders, fit=False)
    return pd.Series(model.predict_proba(X)[:, 1], index=passengers_df.index)


if __name__ == "__main__":
    train_model()
