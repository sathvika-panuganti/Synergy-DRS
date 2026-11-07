"""Train Structure, DS, and DRS XGBoost models for each prioritized dataset."""
from functools import lru_cache
from pathlib import Path
import gc
import json
import sys
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "datasets" / "processed" / "signature_features"
MODEL_FILE = ROOT / "backend" / "models" / "xgboost_precedence_ensemble.joblib"
SUMMARY_FILE = ROOT / "results" / "feature_comparison_holdout_summary.csv"
PRECEDENCE = ("DrugComb", "Almanac", "DrugCombDB", "OncologyScreen", "Oneil")
FAMILIES = ("Structure", "DS", "DRS")
TARGET_THRESHOLD = 10.0
FINGERPRINT_BITS = 512
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FINGERPRINT_BITS)
SIGNATURE_FILES = {
    "DS": ROOT / "datasets" / "signatures" / "DEGWithMASK.npy",
    "DRS": ROOT / "datasets" / "signatures" / "DEGsenWithMASK.npy",
}
sys.path.insert(0, str(Path(__file__).resolve().parent))


def canonical_smiles(value):
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(f"Invalid SMILES in signature dataset: {value}")
    return Chem.MolToSmiles(molecule, canonical=True)


def load_signature_map(path):
    loaded = np.load(path, allow_pickle=True)
    source = loaded.item() if loaded.shape == () else loaded.flat[0]
    result = {}
    for smiles, pair in source.items():
        canonical = canonical_smiles(smiles)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"Invalid signature/mask pair for {smiles}")
        vector = np.asarray(pair[0], dtype=np.float32).ravel()
        mask = np.asarray(pair[1], dtype=np.float32).ravel()
        if vector.shape != (165,) or mask.shape != (165,) or not np.isfinite(vector).all() or not np.isfinite(mask).all():
            raise ValueError(f"Invalid 165-value signature and mask for {smiles}")
        result[canonical] = vector
    return result


def load_identity_rows(dataset):
    path = FEATURE_DIR / f"{dataset}_DRS_features.zip"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run datasets/prepare_signature_features.py first.")
    with ZipFile(path) as archive:
        csv_member = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_member) as stream:
            data = pd.read_csv(stream, usecols=["drugname1", "drugname2", "cellline", "synergy_score"])
    data["drug1"] = data["drugname1"].map(canonical_smiles)
    data["drug2"] = data["drugname2"].map(canonical_smiles)
    reverse = data["drug1"] > data["drug2"]
    data["pair1"] = np.where(reverse, data["drug2"], data["drug1"])
    data["pair2"] = np.where(reverse, data["drug1"], data["drug2"])
    data["cellline"] = data["cellline"].astype(str).str.strip()
    data["synergy_score"] = pd.to_numeric(data["synergy_score"], errors="coerce")
    data = data.dropna(subset=["synergy_score", "cellline"])

    # Archives do not carry dose columns. Aggregate repeated measurements of the same
    # pair/cell-line within each source before applying cross-dataset precedence.
    grouped = data.groupby(["pair1", "pair2", "cellline"], sort=False)["synergy_score"].agg(
        score="mean", observations="size", distinct_scores="nunique",
    ).reset_index()
    grouped["dataset"] = dataset
    return grouped


def make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def xgboost_pipeline(feature_frame):
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError("Install xgboost from backend/requirements.txt before training.") from error
    numeric_columns = [column for column in feature_frame.columns if column != "cellline"]
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            ("cellline", make_one_hot(), ["cellline"]),
        ], sparse_threshold=0.0,
    )
    estimator = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def build_features(rows, family, signature_maps, fingerprint_cache):
    drugs1 = rows["pair1"].to_numpy()
    drugs2 = rows["pair2"].to_numpy()
    if family == "Structure":
        matrix = np.empty((len(rows), 2 * FINGERPRINT_BITS), dtype=np.uint8)
        for index, drug in enumerate(pd.unique(np.concatenate([drugs1, drugs2]))):
            if drug not in fingerprint_cache:
                molecule = Chem.MolFromSmiles(drug)
                bits = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(GENERATOR.GetFingerprint(molecule), bits)
                fingerprint_cache[drug] = bits
        for row_index, (drug1, drug2) in enumerate(zip(drugs1, drugs2)):
            matrix[row_index, :FINGERPRINT_BITS] = fingerprint_cache[drug1]
            matrix[row_index, FINGERPRINT_BITS:] = fingerprint_cache[drug2]
        columns = [f"drug{drug}_bit_{bit}" for drug in (1, 2) for bit in range(FINGERPRINT_BITS)]
    else:
        signature_map = signature_maps[family]
        matrix = np.empty((len(rows), 330), dtype=np.float32)
        for row_index, (drug1, drug2) in enumerate(zip(drugs1, drugs2)):
            matrix[row_index, :165] = signature_map[drug1]
            matrix[row_index, 165:] = signature_map[drug2]
        columns = [f"{family}_drug1_{i:03d}" for i in range(1, 166)]
        columns += [f"{family}_drug2_{i:03d}" for i in range(1, 166)]
    features = pd.DataFrame(matrix, columns=columns)
    features["cellline"] = rows["cellline"].astype(str).to_numpy()
    return features


def evaluate(y_true, y_pred):
    try:
        auc = float(roc_auc_score(y_true >= TARGET_THRESHOLD, y_pred))
    except ValueError:
        auc = float("nan")
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "AUC": auc,
    }


def train_and_save():
    signature_maps = {family: load_signature_map(path) for family, path in SIGNATURE_FILES.items()}
    if set(signature_maps["DS"]) != set(signature_maps["DRS"]):
        raise ValueError("DS and DRS signature maps do not contain the same compounds.")

    observed_scores = {}
    selected_counts = {}
    conflicting_keys = set()
    models, domains, metrics_rows, best_by_dataset = {}, {}, [], {}
    fingerprint_cache = {}
    for dataset in PRECEDENCE:
        rows = load_identity_rows(dataset)
        selected_counts[dataset] = len(rows)
        for row in rows.itertuples(index=False):
            key = (row.pair1, row.pair2, str(row.cellline))
            if key in observed_scores:
                if not np.isclose(float(row.score), observed_scores[key]["score"], rtol=0, atol=1e-8):
                    conflicting_keys.add(key)
                observed_scores[key]["other_sources"].append({
                    "dataset": dataset, "score": float(row.score),
                })
            else:
                observed_scores[key] = {
                    "score": float(row.score), "dataset": dataset,
                    "observations": int(row.observations),
                    "distinct_scores": int(row.distinct_scores),
                    "other_sources": [],
                }
        domains[dataset] = {
            "drugs": sorted(set(rows.pair1).union(rows.pair2)),
            "cell_lines": sorted(rows.cellline.unique().tolist()),
            "rows": len(rows),
        }
        y = rows["score"].to_numpy(dtype=np.float32)
        groups = (rows["pair1"].astype(str) + "||" + rows["pair2"].astype(str)).to_numpy()
        train_indices, test_indices = next(GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42,
        ).split(np.zeros(len(rows)), y, groups=groups))
        models[dataset] = {}
        for family in FAMILIES:
            features = build_features(rows, family, signature_maps, fingerprint_cache)
            candidate = xgboost_pipeline(features)
            candidate.fit(features.iloc[train_indices], y[train_indices])
            predicted = candidate.predict(features.iloc[test_indices])
            heldout_metrics = evaluate(y[test_indices], predicted)
            final_model = xgboost_pipeline(features)
            final_model.fit(features, y)
            models[dataset][family] = {
                "model": final_model,
                "metrics": heldout_metrics,
                "feature_columns": list(features.columns),
            }
            metrics_rows.append({
                "dataset": dataset, "feature_family": family, "model": "XGBoost",
                "train_rows": len(train_indices), "test_rows": len(test_indices),
                **heldout_metrics,
            })
            print(f"{dataset} / {family}: holdout RMSE={heldout_metrics['RMSE']:.4f}; rows={len(rows)}", flush=True)
            del candidate, final_model, features
            gc.collect()
        best_by_dataset[dataset] = min(FAMILIES, key=lambda family: models[dataset][family]["metrics"]["RMSE"])

    # Priority selection is first-match in PRECEDENCE order; retain conflicting lower-priority
    # scores for auditability while the API uses only the first source for the observed score.
    bundle = {
        "models": models,
        "metrics": {"model": "XGBoost", "datasets": list(PRECEDENCE)},
        "precedence": list(PRECEDENCE),
        "domains": domains,
        "supported_drugs": sorted(set().union(*(set(domain["drugs"]) for domain in domains.values()))),
        "supported_cell_lines": sorted(set().union(*(set(domain["cell_lines"]) for domain in domains.values()))),
        "best_feature_family": best_by_dataset,
        "observed_scores": observed_scores,
        "source_row_counts": selected_counts,
        "cross_dataset_conflicts": len(conflicting_keys),
        "aggregation": "Mean score per canonical unordered drug pair and cell line within each dataset; then source precedence selects observed score.",
    }
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE, compress=3)
    summary_path = ROOT / "results" / "xgboost_precedence_holdout_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(summary_path, index=False)
    print(f"Saved model bundle: {MODEL_FILE}")
    print(f"Saved holdout summary: {summary_path}")
    print(f"Precedence-resolved exact pair/cell keys: {len(observed_scores)}; cross-dataset score conflicts: {len(conflicting_keys)}")
    print(f"Best family per dataset: {best_by_dataset}")


if __name__ == "__main__":
    train_and_save()
