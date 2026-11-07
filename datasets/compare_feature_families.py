"""Compare Structure, DS, and DRS with matched repeated CV splits.

Run from the repository root:
    python datasets/compare_feature_families.py

This is a classical-model comparison; SynergyX is intentionally not included.
"""
import argparse
import io
import json
import warnings
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr, t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = next((p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
             if (p / "datasets" / "processed" / "signature_features").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Could not locate datasets/processed/signature_features from this folder.")
FEATURE_DIR = ROOT / "datasets" / "processed" / "signature_features"
RESULT_DIR = ROOT / "results"
DATASETS = ("Almanac", "DrugComb", "DrugCombDB", "OncologyScreen", "Oneil")
FAMILIES = ("Structure", "DS", "DRS")
METRICS = ("MSE", "RMSE", "MAE", "R2", "ExplainedVariance", "Pearson", "Spearman", "AUC")
TARGET_THRESHOLD = 10.0
FINGERPRINT_BITS = 512
FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)


def load_archive(dataset, family):
    path = FEATURE_DIR / f"{dataset}_{family}_features.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature archive: {path}. Run datasets/prepare_signature_features.py first.")
    with ZipFile(path) as archive:
        csv_member = next((name for name in archive.namelist() if name.endswith(".csv")), None)
        if csv_member is None or "metadata.json" not in archive.namelist():
            raise ValueError(f"Archive is missing its CSV or metadata: {path}")
        with archive.open(csv_member) as stream:
            frame = pd.read_csv(stream, low_memory=False)
        metadata = json.loads(archive.read("metadata.json"))
    expected = int(metadata["rows_with_both_drug_signatures"])
    if len(frame) != expected:
        raise ValueError(f"Metadata row count does not match {path.name}: {expected} != {len(frame)}")
    return frame


def identity_frame(frame):
    required = ["drugname1", "drugname2", "cellline", "synergy_score"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Feature archive is missing identity/target columns: {sorted(missing)}")
    return frame[required].reset_index(drop=True)


def fingerprint_features(identity):
    matrix = np.zeros((len(identity), 2 * FINGERPRINT_BITS), dtype=np.uint8)
    for row_i, (smiles1, smiles2) in enumerate(zip(identity.drugname1, identity.drugname2)):
        for drug_i, smiles in enumerate((smiles1, smiles2)):
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is None:
                raise ValueError(f"Invalid SMILES in matched row {row_i}, drug {drug_i + 1}: {smiles}")
            DataStructs.ConvertToNumpyArray(
                FINGERPRINT_GENERATOR.GetFingerprint(molecule),
                matrix[row_i, drug_i * FINGERPRINT_BITS:(drug_i + 1) * FINGERPRINT_BITS],
            )
    columns = [f"drug{drug}_bit_{bit}" for drug in (1, 2) for bit in range(FINGERPRINT_BITS)]
    features = pd.DataFrame(matrix, columns=columns)
    features["cellline"] = identity.cellline.astype(str).to_numpy()
    return features


def signature_features(frame, family):
    prefix = f"{family}_drug"
    columns = [c for c in frame.columns if c.startswith(prefix)]
    if len(columns) != 330:
        raise ValueError(f"Expected 330 {family} pair features, found {len(columns)}")
    features = frame[columns].copy()
    features["cellline"] = frame.cellline.astype(str).to_numpy()
    return features


def stratified_splits(y, folds, repeats, seed):
    """Stratify regression targets by quantile bins, falling back to repeated KFold."""
    q = min(10, max(2, len(y) // folds))
    bins = pd.qcut(pd.Series(y), q=q, labels=False, duplicates="drop")
    counts = pd.Series(bins).value_counts()
    if len(counts) >= 2 and int(counts.min()) >= folds:
        cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        return list(cv.split(np.zeros(len(y)), bins)), "repeated_stratified_quantile_bins"
    warnings.warn("Not enough rows in score quantile bins; using repeated KFold without stratification.")
    cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return list(cv.split(np.zeros(len(y)))), "repeated_kfold_fallback"


def holdout_split(y, seed):
    """Create one stratified 80/20 train/test split for a single evaluation run."""
    q = min(10, max(2, len(y) // 20))
    bins = pd.qcut(pd.Series(y), q=q, labels=False, duplicates="drop")
    counts = pd.Series(bins).value_counts()
    stratify = bins if len(counts) >= 2 and int(counts.min()) >= 2 else None
    train_i, test_i = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=seed, stratify=stratify,
    )
    method = "single_stratified_80_20_holdout" if stratify is not None else "single_random_80_20_holdout"
    return [(train_i, test_i)], method


def make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(model_name, feature_columns):
    numeric = [c for c in feature_columns.columns if c != "cellline"]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "LASSO":
        numeric_steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("cellline", make_one_hot(), ["cellline"]),
        ],
        sparse_threshold=0.0,
    )
    if model_name == "LASSO":
        model = Lasso(alpha=0.01, max_iter=30000, tol=0.001)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=42, n_jobs=-1)
    elif model_name == "AdaBoost":
        tree = DecisionTreeRegressor(max_depth=10, random_state=42)
        options = dict(n_estimators=200, learning_rate=0.05, random_state=42)
        try:
            model = AdaBoostRegressor(estimator=tree, **options)
        except TypeError:
            model = AdaBoostRegressor(base_estimator=tree, **options)
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError("Install the training dependencies from datasets/requirements-training.txt") from error
        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", prep), ("model", model)])


def correlations(actual, predicted):
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = float(pearsonr(actual, predicted).statistic)
        spearman = float(spearmanr(actual, predicted).statistic)
    return pearson, spearman


def evaluate_fold(y_test, predicted):
    pearson, spearman = correlations(y_test, predicted)
    try:
        auc = float(roc_auc_score(y_test >= TARGET_THRESHOLD, predicted))
    except ValueError:
        auc = float("nan")
    return {
        "MSE": float(mean_squared_error(y_test, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "MAE": float(mean_absolute_error(y_test, predicted)),
        "R2": float(r2_score(y_test, predicted)),
        "ExplainedVariance": float(explained_variance_score(y_test, predicted)),
        "Pearson": pearson,
        "Spearman": spearman,
        "AUC": auc,
    }


def summarize(fold_results):
    rows = []
    group_cols = ["dataset", "feature_family", "model"]
    for keys, group in fold_results.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["folds_evaluated"] = len(group)
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            if len(values) > 1:
                half_width = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
                row[f"{metric}_ci95_low"] = row[f"{metric}_mean"] - half_width
                row[f"{metric}_ci95_high"] = row[f"{metric}_mean"] + half_width
            else:
                row[f"{metric}_ci95_low"] = float("nan")
                row[f"{metric}_ci95_high"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def compare_dataset(dataset, folds, repeats, seed, model_names, protocol):
    reference = load_archive(dataset, "DS")
    reference_ids = identity_frame(reference)
    y = pd.to_numeric(reference_ids.synergy_score, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"Non-finite synergy target found in {dataset}")
    if protocol == "holdout":
        split_list, split_method = holdout_split(y, seed)
    else:
        split_list, split_method = stratified_splits(y, folds, repeats, seed)
    print(f"{dataset}: {len(y)} matched rows; {len(split_list)} splits ({split_method})", flush=True)
    all_rows = []

    for family in FAMILIES:
        data = reference if family in ("Structure", "DS") else load_archive(dataset, family)
        if not identity_frame(data).equals(reference_ids):
            raise ValueError(f"The {family} rows do not align with DS for {dataset}.")
        if family == "Structure":
            features = fingerprint_features(reference_ids)
        else:
            features = signature_features(data, family)
        feature_columns = list(features.columns)
        y_family = pd.to_numeric(data.synergy_score, errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(y, y_family):
            raise ValueError(f"Target values differ between feature families for {dataset}.")

        for model_name in model_names:
            print(f"  {family} / {model_name}", flush=True)
            for split_i, (train_i, test_i) in enumerate(split_list):
                print(f"    split {split_i + 1}/{len(split_list)}", flush=True)
                model = build_pipeline(model_name, features)
                model.fit(features.iloc[train_i], y[train_i])
                predicted = model.predict(features.iloc[test_i])
                metrics = evaluate_fold(y[test_i], predicted)
                all_rows.append({
                    "dataset": dataset,
                    "feature_family": family,
                    "model": model_name,
                    "split": split_i,
                    "split_method": split_method,
                    "train_rows": len(train_i),
                    "test_rows": len(test_i),
                    **metrics,
                })
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Compare structure, DS, and DRS with matched splits; SynergyX is excluded.")
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="*", choices=("LASSO", "Random Forest", "AdaBoost", "XGBoost"),
                        default=["LASSO", "Random Forest", "AdaBoost", "XGBoost"])
    parser.add_argument("--protocol", choices=("holdout", "cv"), default="holdout",
                        help="Use one stratified 80/20 split, or repeated cross-validation.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2 or args.repeats < 1:
        parser.error("--folds must be at least 2 and --repeats at least 1")

    all_rows = []
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = RESULT_DIR / f"feature_comparison_{args.protocol}_fold_metrics.csv"
    summary_path = RESULT_DIR / f"feature_comparison_{args.protocol}_summary.csv"
    for dataset in args.datasets:
        all_rows.extend(compare_dataset(dataset, args.folds, args.repeats, args.seed, args.models, args.protocol))
        # Persist completed datasets so a later interruption does not lose earlier work.
        fold_results = pd.DataFrame(all_rows)
        fold_results.to_csv(fold_path, index=False)
        summarize(fold_results).to_csv(summary_path, index=False)
    print(f"Saved fold metrics: {fold_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
