"""Train the DRS XGBoost model served by the API, with matched baselines.

The paper's method is the drug-resistance signature on its own, so the final
synergy score comes from a model that reads only the DRS block. Structure-only
and DS-only models are trained in the same run, with the same split and the
same hyper-parameters, so DRS can be compared against them fairly.

Run from the repository root:

    python backend/services/combined_feature_training.py
"""
from pathlib import Path
import gc
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

SERVICES_DIR = Path(__file__).resolve().parent
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

from signature_features import (  # noqa: E402  (path set above)
    FINAL_FAMILY,
    canonical_smiles,
    drug_structure_vector,
    load_signature_map,
    signature_columns,
    structure_columns,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "datasets" / "processed" / "signature_features"
MODEL_FILE = ROOT / "backend" / "models" / "combined_drs_structure_xgboost.joblib"
METRICS_FILE = ROOT / "results" / "combined_model_metrics.csv"
DIAGNOSTICS_FILE = ROOT / "results" / "combined_model_diagnostics.json"
FRONTEND_REPORT = ROOT / "frontend" / "data" / "model_report.json"

PRECEDENCE = ("DrugComb", "Almanac", "DrugCombDB", "OncologyScreen", "Oneil")
FAMILIES = ("Structure", "DS", "DRS")
SCATTER_POINTS = 1500
RESIDUAL_BINS = 41
TOP_FEATURES = 20
HYPERPARAMETERS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
    tree_method="hist", enable_categorical=True,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_identity_rows(dataset):
    """Unique (canonical unordered pair, cell line) rows with the mean synergy score."""
    path = FEATURE_DIR / f"{dataset}_DRS_features.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python datasets/prepare_signature_features.py` first."
        )
    from zipfile import ZipFile
    with ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(member) as stream:
            data = pd.read_csv(stream, usecols=["drugname1", "drugname2", "cellline", "synergy_score"])

    data["drug1"] = data["drugname1"].map(canonical_smiles)
    data["drug2"] = data["drugname2"].map(canonical_smiles)
    data = data.dropna(subset=["drug1", "drug2"])
    reverse = data["drug1"] > data["drug2"]
    data["pair1"] = np.where(reverse, data["drug2"], data["drug1"])
    data["pair2"] = np.where(reverse, data["drug1"], data["drug2"])
    data["cellline"] = data["cellline"].astype(str).str.strip()
    data["synergy_score"] = pd.to_numeric(data["synergy_score"], errors="coerce")
    data = data.dropna(subset=["synergy_score", "cellline"])
    grouped = data.groupby(["pair1", "pair2", "cellline"], sort=False)["synergy_score"].agg(
        score="mean", observations="size", distinct_scores="nunique",
    ).reset_index()
    grouped["dataset"] = dataset
    return grouped


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _signature_block(pairs1, pairs2, signature_map):
    width = 2 * 165
    matrix = np.empty((len(pairs1), width), dtype=np.float32)
    for index, (drug1, drug2) in enumerate(zip(pairs1, pairs2)):
        matrix[index, :165] = signature_map[drug1]
        matrix[index, 165:] = signature_map[drug2]
    return matrix


def _structure_block(pairs1, pairs2):
    width = 2 * len(drug_structure_vector(pairs1[0]))
    matrix = np.empty((len(pairs1), width), dtype=np.float32)
    half = width // 2
    for index, (drug1, drug2) in enumerate(zip(pairs1, pairs2)):
        matrix[index, :half] = drug_structure_vector(drug1)
        matrix[index, half:] = drug_structure_vector(drug2)
    return matrix


def build_features(rows, family, signature_maps, cell_line_categories):
    """Return a frame whose column order matches the serving-time builders exactly."""
    pairs1 = rows["pair1"].to_numpy()
    pairs2 = rows["pair2"].to_numpy()
    if family == "Structure":
        matrix, columns = _structure_block(pairs1, pairs2), structure_columns()
    elif family in ("DS", "DRS"):
        matrix = _signature_block(pairs1, pairs2, signature_maps[family])
        columns = signature_columns(family)
    else:
        raise ValueError(f"Unknown feature family: {family}")
    if matrix.shape[1] != len(columns):
        raise RuntimeError(f"{family}: built {matrix.shape[1]} values for {len(columns)} columns.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{family}: feature matrix contains non-finite values.")
    features = pd.DataFrame(matrix, columns=columns)
    features["cellline"] = pd.Categorical(
        rows["cellline"].astype(str).to_numpy(), categories=cell_line_categories,
    )
    return features


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def make_model():
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError("Install xgboost from backend/requirements.txt before training.") from error
    return XGBRegressor(**HYPERPARAMETERS)


def evaluate(y_true, y_pred):
    """Regression metrics only: MSE, MAE, RMSE, R2 and both rank correlations."""
    from scipy.stats import spearmanr

    mse = float(mean_squared_error(y_true, y_pred))
    constant = float(np.std(y_pred)) == 0.0
    return {
        "MSE": mse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float("nan") if constant else float(np.corrcoef(y_true, y_pred)[0, 1]),
        "Spearman": float("nan") if constant else float(spearmanr(y_true, y_pred).statistic),
    }


def _scatter_sample(y_true, y_pred, seed=42):
    count = len(y_true)
    if count > SCATTER_POINTS:
        index = np.random.default_rng(seed).choice(count, SCATTER_POINTS, replace=False)
    else:
        index = np.arange(count)
    return [[round(float(y_true[i]), 3), round(float(y_pred[i]), 3)] for i in index]


def _residual_histogram(y_true, y_pred):
    residuals = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    limit = float(np.percentile(np.abs(residuals), 99)) or 1.0
    counts, edges = np.histogram(np.clip(residuals, -limit, limit), bins=RESIDUAL_BINS)
    return {
        "counts": counts.tolist(),
        "edges": [round(float(edge), 4) for edge in edges],
        "mean": round(float(residuals.mean()), 4),
        "std": round(float(residuals.std()), 4),
    }


def _feature_importance(model, columns):
    booster = model.get_booster()
    gains = booster.get_score(importance_type="gain")
    # XGBoost reports f<index> when the frame is passed as a numeric matrix and
    # the real name when feature names survive; handle both.
    named = {}
    for key, value in gains.items():
        if key.startswith("f") and key[1:].isdigit() and int(key[1:]) < len(columns):
            named[columns[int(key[1:])]] = float(value)
        else:
            named[key] = float(value)
    total = sum(named.values()) or 1.0
    top = sorted(named.items(), key=lambda item: item[1], reverse=True)[:TOP_FEATURES]

    def group_of(name):
        """Split DRS values from the drug-1 / drug-2 halves and the cell line."""
        if name.startswith("cellline"):
            return "Cell line"
        if name.startswith("DRS_drug1") or name.startswith("DS_drug1"):
            return "Drug 1 signature"
        if name.startswith("DRS_drug2") or name.startswith("DS_drug2"):
            return "Drug 2 signature"
        return "Structure"

    groups = {}
    for name, value in named.items():
        groups[group_of(name)] = groups.get(group_of(name), 0.0) + value
    return {
        "top": [{"feature": name, "gain": round(value, 4), "share": round(value / total, 5),
                 "group": group_of(name)}
                for name, value in top],
        "group_share": {name: round(value / total, 5) for name, value in groups.items()},
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_and_save(datasets=PRECEDENCE):
    signature_maps = {family: load_signature_map(family) for family in ("DS", "DRS")}
    if set(signature_maps["DS"]) != set(signature_maps["DRS"]):
        raise ValueError("The DS and DRS signature maps do not contain the same compounds.")

    observed_scores, selected_counts, conflicting_keys = {}, {}, set()
    models, domains, metrics_rows, diagnostics = {}, {}, [], {}

    for dataset in datasets:
        rows = load_identity_rows(dataset)
        selected_counts[dataset] = len(rows)
        unsupported = {drug for drug in set(rows.pair1).union(rows.pair2)
                       if drug not in signature_maps["DRS"]}
        if unsupported:
            rows = rows[~rows.pair1.isin(unsupported) & ~rows.pair2.isin(unsupported)].reset_index(drop=True)
            print(f"{dataset}: dropped {len(unsupported)} compounds without a DRS signature", flush=True)
        if rows.empty:
            raise ValueError(f"{dataset}: no rows left after DRS signature matching.")

        for row in rows.itertuples(index=False):
            key = (row.pair1, row.pair2, str(row.cellline))
            if key in observed_scores:
                if not np.isclose(float(row.score), observed_scores[key]["score"], rtol=0, atol=1e-8):
                    conflicting_keys.add(key)
                observed_scores[key]["other_sources"].append({"dataset": dataset, "score": float(row.score)})
            else:
                observed_scores[key] = {
                    "score": float(row.score), "dataset": dataset,
                    "observations": int(row.observations),
                    "distinct_scores": int(row.distinct_scores),
                    "other_sources": [],
                }

        cell_line_categories = sorted(rows.cellline.astype(str).unique().tolist())
        domains[dataset] = {
            "drugs": sorted(set(rows.pair1).union(rows.pair2)),
            "cell_lines": cell_line_categories,
            "rows": len(rows),
        }

        y = rows["score"].to_numpy(dtype=np.float32)
        groups = (rows["pair1"].astype(str) + "||" + rows["pair2"].astype(str)).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_index, test_index = next(splitter.split(np.zeros(len(rows)), y, groups=groups))
        # Drug pairs never appear in both sides of the split, so a pair seen in
        # training cannot leak its score into the held-out estimate.
        assert not set(groups[train_index]) & set(groups[test_index])

        models[dataset] = {}
        for family in FAMILIES:
            features = build_features(rows, family, signature_maps, cell_line_categories)
            holdout_model = make_model()
            holdout_model.fit(features.iloc[train_index], y[train_index])
            predicted = holdout_model.predict(features.iloc[test_index])
            metrics = evaluate(y[test_index], predicted)

            # The served model is refit on every row of the dataset; the metrics
            # above come only from the held-out pairs.
            final_model = make_model()
            final_model.fit(features, y)
            models[dataset][family] = {
                "model": final_model,
                "metrics": metrics,
                "feature_columns": list(features.columns),
                "cell_line_categories": cell_line_categories,
            }
            metrics_rows.append({
                "dataset": dataset, "feature_family": family, "model": "XGBoost",
                "n_features": features.shape[1] - 1,
                "train_rows": int(len(train_index)), "test_rows": int(len(test_index)),
                **metrics,
            })
            if family == FINAL_FAMILY:
                diagnostics[dataset] = {
                    "scatter": _scatter_sample(y[test_index], predicted),
                    "residuals": _residual_histogram(y[test_index], predicted),
                    "importance": _feature_importance(final_model, list(features.columns)),
                    "test_rows": int(len(test_index)),
                    "score_range": [round(float(y.min()), 3), round(float(y.max()), 3)],
                }
            print(f"{dataset} / {family}: RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} "
                  f"R2={metrics['R2']:.4f} ({features.shape[1] - 1} features)", flush=True)
            del features, holdout_model, final_model
            gc.collect()

    bundle = {
        "final_family": FINAL_FAMILY,
        "families": list(FAMILIES),
        "models": models,
        "precedence": list(PRECEDENCE),
        "domains": domains,
        "supported_drugs": sorted(set().union(*(set(d["drugs"]) for d in domains.values()))),
        "supported_cell_lines": sorted(set().union(*(set(d["cell_lines"]) for d in domains.values()))),
        "observed_scores": observed_scores,
        "source_row_counts": selected_counts,
        "cross_dataset_conflicts": len(conflicting_keys),
        "hyperparameters": HYPERPARAMETERS,
        "protocol": ("Grouped 80/20 holdout split by canonical unordered drug pair per dataset "
                     "(random_state=42); served models are refit on all rows of their dataset."),
        "aggregation": ("Mean score per canonical unordered drug pair and cell line within each dataset; "
                        "cross-dataset precedence selects the recorded score."),
    }
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE, compress=3)

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(METRICS_FILE, index=False)

    report = {
        "protocol": bundle["protocol"],
        "final_family": FINAL_FAMILY,
        "hyperparameters": {k: v for k, v in HYPERPARAMETERS.items() if k != "n_jobs"},
        "metrics": metrics_rows,
        "diagnostics": diagnostics,
    }
    with DIAGNOSTICS_FILE.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    FRONTEND_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with FRONTEND_REPORT.open("w", encoding="utf-8") as stream:
        json.dump(report, stream)

    print(f"\nSaved model bundle:  {MODEL_FILE}")
    print(f"Saved metrics:       {METRICS_FILE}")
    print(f"Saved diagnostics:   {DIAGNOSTICS_FILE}")
    print(f"Recorded pair/cell keys: {len(observed_scores)}; cross-dataset conflicts: {len(conflicting_keys)}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the DRS XGBoost model and its baselines.")
    parser.add_argument("--datasets", nargs="*", choices=list(PRECEDENCE), default=list(PRECEDENCE))
    arguments = parser.parse_args()
    train_and_save(tuple(arguments.datasets))
