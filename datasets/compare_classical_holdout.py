"""Run one leakage-aware grouped holdout comparison for classical models.

This deliberately uses one 80/20 split per dataset, grouped by canonical drug pair.
SynergyX is excluded as requested. Run from the repository root:
    python datasets/compare_classical_holdout.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.multi_dataset_xgboost_training import (
    ROOT, PRECEDENCE, FAMILIES, SIGNATURE_FILES, TARGET_THRESHOLD,
    load_identity_rows, load_signature_map, build_features, xgboost_pipeline,
)

MODELS = ("LASSO", "Random Forest", "AdaBoost", "XGBoost")
RESULTS = ROOT / "results"
METRICS_JSON = ROOT / "frontend" / "data" / "metrics.json"


def one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def classical_pipeline(name, features):
    numeric = [c for c in features.columns if c != "cellline"]
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name == "LASSO":
        steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("cellline", one_hot(), ["cellline"]),
    ], sparse_threshold=0.0)
    if name == "LASSO":
        estimator = Lasso(alpha=0.01, max_iter=5000, tol=0.01, selection="random", random_state=42)
    elif name == "Random Forest":
        estimator = RandomForestRegressor(n_estimators=50, max_depth=20,
            max_features=0.25, min_samples_leaf=2, random_state=42, n_jobs=-1)
    elif name == "AdaBoost":
        tree = DecisionTreeRegressor(max_depth=4, random_state=42)
        kwargs = dict(n_estimators=25, learning_rate=0.05, random_state=42)
        try:
            estimator = AdaBoostRegressor(estimator=tree, **kwargs)
        except TypeError:
            estimator = AdaBoostRegressor(base_estimator=tree, **kwargs)
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", prep), ("model", estimator)])


def metrics(actual, predicted):
    if len(actual) > 1 and np.std(actual) and np.std(predicted):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson = float(pearsonr(actual, predicted).statistic)
            spearman = float(spearmanr(actual, predicted).statistic)
    else:
        pearson = spearman = float("nan")
    try:
        auc = float(roc_auc_score(actual >= TARGET_THRESHOLD, predicted))
    except ValueError:
        auc = float("nan")
    mse = float(mean_squared_error(actual, predicted))
    return {
        "MSE": mse, "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
        "ExplainedVariance": float(explained_variance_score(actual, predicted)),
        "Pearson": pearson, "Spearman": spearman, "AUC": auc,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    (ROOT / "frontend" / "data").mkdir(parents=True, exist_ok=True)
    signatures = {family: load_signature_map(path) for family, path in SIGNATURE_FILES.items()}
    if set(signatures["DS"]) != set(signatures["DRS"]):
        raise ValueError("DS and DRS maps must cover the same compounds")
    fingerprints = {}
    result_rows = []
    csv_path = RESULTS / "classical_grouped_holdout_summary.csv"
    for dataset in PRECEDENCE:
        data = load_identity_rows(dataset)
        y = data.score.to_numpy(dtype=np.float32)
        groups = (data.pair1.astype(str) + "||" + data.pair2.astype(str)).to_numpy()
        train_i, test_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
            random_state=42).split(np.zeros(len(data)), y, groups=groups))
        print(f"{dataset}: {len(data):,} aggregated pair/cell rows; one grouped holdout; "
              f"train={len(train_i):,}, test={len(test_i):,}", flush=True)
        for family in FAMILIES:
            features = build_features(data, family, signatures, fingerprints)
            for model_name in MODELS:
                estimator = (xgboost_pipeline(features) if model_name == "XGBoost"
                    else classical_pipeline(model_name, features))
                estimator.fit(features.iloc[train_i], y[train_i])
                pred = estimator.predict(features.iloc[test_i])
                row = {
                    "dataset": dataset, "feature_family": family,
                    "model": model_name, "train_rows": len(train_i),
                    "test_rows": len(test_i), "split": "grouped_80_20_seed_42",
                    **metrics(y[test_i], pred),
                }
                result_rows.append(row)
                pd.DataFrame(result_rows).to_csv(csv_path, index=False)
                print(f"  {family} / {model_name}: RMSE={row['RMSE']:.4f}", flush=True)
                del estimator, pred
            del features
    frame = pd.DataFrame(result_rows)
    frame.to_csv(csv_path, index=False)
    json_rows = []
    for row in result_rows:
        json_rows.append({
            "dataset": row["dataset"], "feature_family": row["feature_family"],
            "model": row["model"], "train_rows": row["train_rows"],
            "test_rows": row["test_rows"],
            **{key.lower() + "_mean": (None if not np.isfinite(row[key]) else row[key])
               for key in ("MSE", "RMSE", "MAE", "R2", "ExplainedVariance", "Pearson", "Spearman", "AUC")},
        })
    report = {
        "evaluation": {
            "protocol": "One grouped 80/20 holdout split by canonical unordered drug pair per dataset (random_state=42).",
            "repeats": 1, "folds": 1,
            "coverage": "LASSO, Random Forest, AdaBoost, and XGBoost evaluated for Structure, DS, and DRS on all five datasets.",
            "models": list(MODELS), "datasets": list(PRECEDENCE),
            "feature_families": list(FAMILIES),
            "model_settings": {
                "LASSO": "alpha=0.01, max_iter=5000, tol=0.01",
                "Random Forest": "50 trees, max_depth=20, max_features=0.25, min_samples_leaf=2",
                "AdaBoost": "25 estimators, learning_rate=0.05, decision-tree max_depth=4",
                "XGBoost": "300 estimators, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8",
            },
            "precedence": "DrugComb > Almanac > DrugCombDB > OncologyScreen > Oneil",
            "score_handling": "Repeated scores for a canonical unordered drug pair and cell line are averaged within each source. The dashboard benchmarks each dataset separately.",
            "synergy_threshold": 10.0,
            "selection": "Dashboard final predictions use the XGBoost family with the lowest holdout RMSE for the selected dataset.",
            "limitations": "Single holdout estimates vary by split. SynergyX and repeated cross-validation are not included. This project uses categorical cell-line encoding rather than the author's 24,474-value cell-line graph vector, so results are not an exact reproduction of the paper.",
        },
        "metrics": json_rows,
    }
    METRICS_JSON.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(frame)} rows to {csv_path}", flush=True)
    print(f"Updated dashboard metrics: {METRICS_JSON}", flush=True)


if __name__ == "__main__":
    main()
