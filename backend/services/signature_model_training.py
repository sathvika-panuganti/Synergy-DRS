"""Train the deployable Oneil DRS + XGBoost model used by the API."""
from pathlib import Path
import json
import sys
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
FEATURE_ARCHIVE = ROOT / "datasets" / "processed" / "signature_features" / "Oneil_DRS_features.zip"
HOLDOUT_SUMMARY = ROOT / "results" / "feature_comparison_holdout_summary.csv"
MODEL_FILE = ROOT / "backend" / "models" / "oneil_drs_xgboost.joblib"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signature_features import load_drs_signatures  # noqa: E402


def make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_training_frame():
    if not FEATURE_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Missing {FEATURE_ARCHIVE}. Run `python datasets/prepare_signature_features.py --datasets Oneil` first."
        )
    with ZipFile(FEATURE_ARCHIVE) as archive:
        csv_member = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_member) as stream:
            data = pd.read_csv(stream, low_memory=False)
        metadata = json.loads(archive.read("metadata.json"))
    if len(data) != int(metadata["rows_with_both_drug_signatures"]):
        raise ValueError("Oneil DRS archive row count does not match its metadata.")
    numeric_columns = [column for column in data.columns if column.startswith("DRS_drug")]
    if len(numeric_columns) != 330:
        raise ValueError(f"Expected 330 Oneil DRS features, found {len(numeric_columns)}.")
    data["synergy_score"] = pd.to_numeric(data["synergy_score"], errors="coerce")
    data = data.dropna(subset=numeric_columns + ["cellline", "synergy_score"]).reset_index(drop=True)
    return data, numeric_columns


def load_holdout_metrics():
    if not HOLDOUT_SUMMARY.exists():
        return {}
    summary = pd.read_csv(HOLDOUT_SUMMARY)
    selected = summary[
        (summary["dataset"] == "Oneil")
        & (summary["feature_family"] == "DRS")
        & (summary["model"] == "XGBoost")
    ]
    if selected.empty:
        return {}
    row = selected.iloc[0]
    return {key.removesuffix("_mean"): float(row[key]) for key in ("RMSE_mean", "MAE_mean", "R2_mean", "AUC_mean")}


def train_and_save():
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError("Install xgboost from datasets/requirements-training.txt before training.") from error

    data, numeric_columns = load_training_frame()
    x = data[numeric_columns + ["cellline"]].copy()
    y = data["synergy_score"].astype(float)
    prep = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            ("cellline", make_one_hot(), ["cellline"]),
        ],
        sparse_threshold=0.0,
    )
    model = Pipeline([
        ("preprocess", prep),
        ("model", XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(x, y)
    bundle = {
        "model": model,
        "metrics": {
            "dataset": "Oneil",
            "model": "XGBoost with DRS signatures",
            "training_rows": len(data),
            "evaluation_protocol": "single stratified 80/20 holdout",
            "holdout_metrics": load_holdout_metrics(),
        },
        "feature_columns": numeric_columns + ["cellline"],
        "supported_drugs": sorted(load_drs_signatures()),
        "supported_cell_lines": sorted(data["cellline"].astype(str).unique()),
    }
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE, compress=3)
    print(f"Saved DRS model: {MODEL_FILE}")
    print(f"Training rows: {len(data)}; supported DRS compounds: {len(bundle['supported_drugs'])}")
    print(f"Oneil holdout metrics: {bundle['metrics']['holdout_metrics']}")


if __name__ == "__main__":
    train_and_save()
