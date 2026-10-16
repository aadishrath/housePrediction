"""Train status-aware national home price models from Realtor listing data."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from xgboost import XGBRegressor
from model_components import TorchMLPRegressor, inverse_log_price

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "house_data.csv"
MODEL_DIR = ROOT / "models"
FEATURE_DIR = ROOT / "features"
TARGET = "price"
STATUS = "status"
FEATURES = [STATUS, "bed", "bath", "house_size", "acre_lot", "state", "city", "zip_code"]
NUMERIC_FEATURES = ["bed", "bath", "house_size", "acre_lot"]
CATEGORICAL_FEATURES = [STATUS, "state", "city", "zip_code"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
PRICE_MIN, PRICE_MAX = 10_000, 100_000_000

INPUT_SCHEMA = [
    {"id": STATUS, "label": "Price context", "type": "select", "required": True, "options": [
        {"value": "sold", "label": "Completed sale price"},
        {"value": "for_sale", "label": "For-sale asking price"},
        {"value": "ready_to_build", "label": "Ready-to-build asking price"}],
        "help": "Choose the kind of price you want estimated. The model is trained on all three data statuses."},
    {"id": "bed", "label": "Bedrooms", "type": "integer", "min": 0, "max": 30, "step": 1, "required": True, "required_unless": "ready_to_build", "help": "Whole number; optional for ready-to-build listings."},
    {"id": "bath", "label": "Bathrooms", "type": "decimal", "min": 0, "max": 30, "step": 0.5, "required": True, "required_unless": "ready_to_build", "help": "Half-bath increments; optional for ready-to-build listings."},
    {"id": "house_size", "label": "Living area (sq ft)", "type": "integer", "min": 1, "max": 50000, "step": 1, "required": True, "required_unless": "ready_to_build", "help": "Finished interior area; optional for ready-to-build listings."},
    {"id": "acre_lot", "label": "Lot size (acres)", "type": "decimal", "min": 0, "max": 1000, "step": 0.01, "required": False, "help": "Optional. Leave blank if unknown."},
    {"id": "state", "label": "State", "type": "text", "min_length": 2, "max_length": 2, "required": True, "help": "Two-letter state abbreviation, such as CA."},
    {"id": "city", "label": "City", "type": "text", "min_length": 1, "max_length": 100, "required": True},
    {"id": "zip_code", "label": "ZIP code", "type": "text", "pattern": r"^\d{5}$", "required": True, "help": "Enter a five-digit ZIP code."},
]

FEATURE_GUIDE = {
    "included": [
        {"id": STATUS, "reason": "Selects the price meaning: completed sale price or asking/list price."},
        {"id": "bed", "reason": "Bedroom count describes the property's layout."},
        {"id": "bath", "reason": "Bathroom count captures another basic size and utility signal."},
        {"id": "house_size", "reason": "Interior square footage is a primary property-size measure."},
        {"id": "acre_lot", "reason": "Parcel size adds land context; missing values are imputed."},
        {"id": "state", "reason": "Broad geographic differences in observed prices."},
        {"id": "city", "reason": "Local price variation; encoded with training-only target statistics."},
        {"id": "zip_code", "reason": "Fine-grained location; encoded as a category rather than a number."},
    ],
    "excluded": [
        {"id": "brokered_by", "reason": "Broker/agent identifier is not a property characteristic and can memorize records."},
        {"id": "street", "reason": "Street/record identifier is high-cardinality and not reliably available for user inputs."},
        {"id": "prev_sold_date", "reason": "Previous sale date is not the current listing/sale observation date; it is incomplete and contains invalid values."},
        {"id": "price", "reason": "This is the target being estimated; its meaning depends on status."},
    ],
}
warnings.filterwarnings("ignore", category=UserWarning)


def atomic_joblib_dump(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(value, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item) if np.isfinite(item) else None
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def make_pipeline(regressor, scale=False) -> TransformedTargetRegressor:
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))]), NUMERIC_FEATURES),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", TargetEncoder(target_type="continuous", smooth="auto", cv=3, shuffle=True, random_state=RANDOM_STATE))]), CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)
    steps = [("features", preprocessing)]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("regressor", regressor))
    estimator = Pipeline(steps)
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=inverse_log_price)


def score_model(actual: pd.Series, predicted: np.ndarray) -> dict:
    errors = np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    rank = min(len(errors), int(np.ceil((len(errors) + 1) * 0.80)))
    radius = float(np.partition(np.abs(errors), rank - 1)[rank - 1])
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "median_ae": float(median_absolute_error(actual, predicted)),
        "mape": float(mean_absolute_percentage_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mean_error": float(np.mean(np.asarray(predicted) - np.asarray(actual))),
        "validation_band_radius_80": radius,
        "validation_band_coverage": float(np.mean(np.abs(errors) <= radius)),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    # Read only needed columns; dtype conversion keeps a multi-million row file manageable.
    raw_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    needed = FEATURES + [TARGET]
    missing = sorted(set(needed) - set(raw_columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    data = pd.read_csv(DATA_PATH, usecols=needed, low_memory=False)
    initial_rows = len(data)
    data[STATUS] = data[STATUS].astype("string").str.strip().str.lower()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    for feature in NUMERIC_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce").astype("float32")
    for feature in ["state", "city", "zip_code"]:
        data[feature] = data[feature].astype("string").str.strip()
        data.loc[data[feature].isin(["", "nan", "<NA>"]), feature] = pd.NA
    data["state"] = data["state"].str.upper()
    data["zip_code"] = data["zip_code"].str.replace(r"\.0$", "", regex=True).str.zfill(5)
    data["city"] = data["city"].str.title()
    data = data.replace([np.inf, -np.inf], np.nan)
    # Zero beds/baths can be a real studio/no-bath property, so preserve them.
    # A zero interior area is not a usable home size and is treated as missing.
    data.loc[data["house_size"] == 0, "house_size"] = np.nan
    valid = data[TARGET].between(PRICE_MIN, PRICE_MAX) & data[STATUS].isin(["sold", "for_sale", "ready_to_build"])
    valid &= data["state"].notna() & data["city"].notna() & data["zip_code"].str.fullmatch(r"\d{5}", na=False)
    valid &= data["bed"].isna() | data["bed"].between(0, 30)
    valid &= data["bath"].isna() | data["bath"].between(0, 30)
    valid &= data["house_size"].isna() | data["house_size"].between(1, 50_000)
    valid &= data["acre_lot"].isna() | data["acre_lot"].between(0, 1_000)
    data = data.loc[valid, needed].reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid training rows remain after data quality filtering.")
    data["zip_code"] = data["zip_code"].astype(str)
    data["state"] = data["state"].astype(str)
    data["city"] = data["city"].astype(str)
    X = data[FEATURES]
    y = data[TARGET].astype("float32")
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data[STATUS])

    cuda_available = bool(torch.cuda.is_available())
    models = {
        "Linear Regression": (LinearRegression(), "cpu", True),
        # Epsilon-insensitive SGD is a scalable linear SVR objective for millions of rows.
        "Linear SVR": (SGDRegressor(loss="epsilon_insensitive", penalty="l2", alpha=1e-4, max_iter=20, tol=None, average=True, random_state=RANDOM_STATE), "cpu", True),
        "Neural Network": (TorchMLPRegressor(device="cuda" if cuda_available else "cpu", epochs=12, batch_size=16384, random_state=RANDOM_STATE), "cuda" if cuda_available else "cpu", True),
        "Gradient Boosting": (HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=3.0, random_state=RANDOM_STATE), "cpu", False),
        "Random Forest": (RandomForestRegressor(n_estimators=32, max_depth=14, min_samples_leaf=20, max_features=1.0, n_jobs=4, random_state=RANDOM_STATE), "cpu", False),
        "XGBoost": (XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cuda" if cuda_available else "cpu", n_estimators=220, max_depth=8, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, min_child_weight=5, reg_lambda=8.0, n_jobs=1, random_state=RANDOM_STATE, verbosity=0), "cuda" if cuda_available else "cpu", False),
    }
    metrics = {}
    for label, (regressor, requested_device, scale) in models.items():
        pipeline = make_pipeline(regressor, scale)
        device = requested_device
        print(f"Training {label} ({device.upper()}) on {len(x_train):,} rows; validating on {len(x_valid):,}...")
        try:
            pipeline.fit(x_train, y_train)
        except Exception as error:
            if device != "cuda" or label not in {"XGBoost", "Neural Network"}:
                raise
            print(f"CUDA training failed ({error}); retrying {label} on CPU.")
            regressor.set_params(device="cpu")
            pipeline = make_pipeline(regressor, scale)
            pipeline.fit(x_train, y_train)
            device = "cpu"
        fitted_estimator = pipeline.regressor_.named_steps["regressor"]
        device = getattr(fitted_estimator, "device_", device)
        predictions = pipeline.predict(x_valid)
        result = score_model(y_valid, predictions)
        result["by_status"] = {}
        for status, idx in x_valid.groupby(STATUS).groups.items():
            result["by_status"][status] = score_model(y_valid.loc[idx], predictions[x_valid.index.get_indexer(idx)])
        result.update({"training_device": device, "features": FEATURES, "best_parameters": regressor.get_params(deep=False)})
        # Keep the holdout strictly for model comparison; deployment artifacts see every retained row.
        print(f"  Refit {label} on all {len(X):,} retained rows for deployment...")
        pipeline = make_pipeline(regressor, scale)
        pipeline.fit(X, y)
        # The serialized model is portable for CPU-based Flask inference.
        if label == "XGBoost":
            pipeline.regressor_.named_steps["regressor"].set_params(device="cpu")
        atomic_joblib_dump(pipeline, MODEL_DIR / f"{'_'.join(label.lower().split())}.pkl")
        metrics[label] = result
        print(f"  MAE ${result['mae']:,.0f} · RMSE ${result['rmse']:,.0f} · R² {result['r2']:.3f} · 80% band ±${result['validation_band_radius_80']:,.0f}")

    status_counts = data[STATUS].value_counts().to_dict()
    metadata = {
        "source_file": DATA_PATH.name,
        "source_url": "https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset",
        "raw_columns": raw_columns,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "rows_raw": initial_rows,
        "rows_used": int(len(data)),
        "deployment_training_rows": int(len(data)),
        "rows_excluded": int(initial_rows - len(data)),
        "status_counts": {key: int(value) for key, value in status_counts.items()},
        "target_definition": "price conditioned on status: sold=completed sale price; for_sale and ready_to_build=asking/list price",
        "price_filter_usd": [PRICE_MIN, PRICE_MAX],
        "validation_fraction": TEST_SIZE,
        "validation_split": "random stratified holdout by status; not a time-based or geographic external validation",
        "random_state": RANDOM_STATE,
        "target_transform": "log1p",
        "gpu_available": cuda_available,
        "note": "The source has no current observation date; predictions are status-conditioned dataset estimates, not current appraisals.",
    }
    save_json(FEATURE_DIR / "input_schema.json", INPUT_SCHEMA)
    save_json(FEATURE_DIR / "modeling_notes.json", FEATURE_GUIDE)
    save_json(FEATURE_DIR / "selected_features.json", {name: FEATURES for name in models})
    save_json(MODEL_DIR / "model_metrics.json", metrics)
    save_json(MODEL_DIR / "training_metadata.json", metadata)
    best = min(metrics, key=lambda name: metrics[name]["mae"])
    print(f"Rows used: {len(data):,}/{initial_rows:,}. Best holdout MAE: {best} (${metrics[best]['mae']:,.0f}).")


if __name__ == "__main__":
    main()
