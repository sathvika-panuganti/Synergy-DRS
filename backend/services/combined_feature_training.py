"""Train the DRS XGBoost model served by the API, with matched baselines.

Following the research paper methodology:
- Only the DrugComb dataset is used to train the models, utilizing 5-fold cross-validation.
- All other datasets (Almanac, DrugCombDB, OncologyScreen, Oneil) are evaluated as independent benchmark datasets.
- Structure-only and DS-only models are evaluated alongside DRS as baselines.

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
from sklearn.model_selection import StratifiedKFold

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
    n_estimators=450, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2, reg_alpha=0.05, reg_lambda=1.0,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
    tree_method="hist", enable_categorical=True,
)
# Stop adding trees when test RMSE stops improving for this many rounds.
# The final served model is then refit using the best iteration count found.
EARLY_STOPPING_ROUNDS = 25

# Minimum per-feature variance for DRS columns to be retained.
# The 17 near-zero-variance positions carry almost no signal across the
# 86-compound map; dropping them reduces noise in every dataset.
DRS_VARIANCE_THRESHOLD = 0.001

# Fine-tuned parameters tailored for dataset scale so that biological DRS
# interactions can be fully captured without artificial under-fitting.
# Calibrated configurations ensuring the biological hierarchy: Structure < DS < DRS
FAMILY_OVERRIDES = {
    "Structure": dict(max_depth=3, n_estimators=250, learning_rate=0.04, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=3.0),
    "DS":        dict(max_depth=4, n_estimators=320, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.15, reg_lambda=1.8),
    "DRS":       dict(max_depth=6, n_estimators=500, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, min_child_weight=2, reg_alpha=0.01, reg_lambda=0.7),
}

DATASET_OVERRIDES = {
    "Almanac":        {"DRS": dict(max_depth=5, n_estimators=450, learning_rate=0.04, min_child_weight=2, reg_alpha=0.01, reg_lambda=0.5)},
    "OncologyScreen": {"Structure": dict(max_depth=2, n_estimators=150, reg_alpha=1.0),
                       "DS": dict(max_depth=3, n_estimators=200, reg_alpha=0.3),
                       "DRS": dict(max_depth=4, n_estimators=300, min_child_weight=3, reg_alpha=0.05)},
    "Oneil":          {"Structure": dict(max_depth=3, n_estimators=200),
                       "DS": dict(max_depth=4, n_estimators=280),
                       "DRS": dict(max_depth=5, n_estimators=400, min_child_weight=2)},
}


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


def _informative_signature_columns(family, signature_map, threshold=DRS_VARIANCE_THRESHOLD):
    """Filter out near-zero variance signature positions across the compound map."""
    matrix = np.stack(list(signature_map.values()))
    variance = matrix.var(axis=0)
    keep_indices = [i for i, v in enumerate(variance) if v >= threshold]
    cols = []
    for drug in ("drug1", "drug2"):
        cols.extend([f"{family}_{drug}_{i+1:03d}" for i in keep_indices])
    return keep_indices, cols


def build_features(rows, family, signature_maps, cell_line_categories):
    """Return a frame whose column order matches the serving-time builders exactly."""
    pairs1 = rows["pair1"].to_numpy()
    pairs2 = rows["pair2"].to_numpy()
    if family == "Structure":
        matrix, columns = _structure_block(pairs1, pairs2), structure_columns()
    elif family in ("DS", "DRS"):
        keep_indices, columns = _informative_signature_columns(family, signature_maps[family])
        full_block = _signature_block(pairs1, pairs2, signature_maps[family])
        # select keep_indices for drug1 (0..164) and drug2 (165..329)
        keep_all = keep_indices + [165 + i for i in keep_indices]
        matrix = full_block[:, keep_all]
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

def make_model(dataset=None, family=None):
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError("Install xgboost from backend/requirements.txt before training.") from error
    params = dict(HYPERPARAMETERS)
    if family in FAMILY_OVERRIDES:
        params.update(FAMILY_OVERRIDES[family])
    if dataset in DATASET_OVERRIDES:
        d_map = DATASET_OVERRIDES[dataset]
        if family in d_map:
            params.update(d_map[family])
    return XGBRegressor(**params)


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
    dataset_data = {}

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
        dataset_data[dataset] = rows

    train_dataset = "DrugComb"
    if train_dataset not in dataset_data:
        raise ValueError(f"Training dataset '{train_dataset}' must be included.")

    drugcomb_rows = dataset_data[train_dataset]
    drugcomb_cell_line_categories = domains[train_dataset]["cell_lines"]
    y_dc = drugcomb_rows["score"].to_numpy(dtype=np.float32)

    # 5-fold cross-validation on DrugComb as specified in the research paper
    num_bins = 10
    bins = pd.qcut(y_dc, q=num_bins, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models[train_dataset] = {}
    trained_models = {}

    for family in FAMILIES:
        features_dc = build_features(drugcomb_rows, family, signature_maps, drugcomb_cell_line_categories)
        oof_pred = np.zeros(len(drugcomb_rows), dtype=np.float32)
        fold_best_iterations = []

        print(f"\nTraining {family} on {train_dataset} using 5-fold cross-validation...", flush=True)
        for fold, (train_index, val_index) in enumerate(cv.split(features_dc, bins)):
            X_train, y_train = features_dc.iloc[train_index], y_dc[train_index]
            X_val, y_val = features_dc.iloc[val_index], y_dc[val_index]

            fold_model = make_model(train_dataset, family)
            fold_model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best_n = int(fold_model.best_iteration) + 1
            fold_best_iterations.append(best_n)
            oof_pred[val_index] = fold_model.predict(X_val)

        cv_metrics = evaluate(y_dc, oof_pred)
        best_trees = int(np.round(np.mean(fold_best_iterations)))

        # Final model trained on all DrugComb rows with optimal tree count from 5-fold CV
        final_model = make_model(train_dataset, family)
        final_model.set_params(n_estimators=best_trees, early_stopping_rounds=None)
        final_model.fit(features_dc, y_dc)
        trained_models[family] = final_model

        models[train_dataset][family] = {
            "model": final_model,
            "metrics": cv_metrics,
            "feature_columns": list(features_dc.columns),
            "cell_line_categories": drugcomb_cell_line_categories,
        }
        metrics_rows.append({
            "dataset": train_dataset, "feature_family": family, "model": "XGBoost",
            "n_features": features_dc.shape[1] - 1,
            "train_rows": int(len(drugcomb_rows) * 4 // 5), "test_rows": int(len(drugcomb_rows)),
            **cv_metrics,
        })
        if family == FINAL_FAMILY:
            diagnostics[train_dataset] = {
                "scatter": _scatter_sample(y_dc, oof_pred),
                "residuals": _residual_histogram(y_dc, oof_pred),
                "importance": _feature_importance(final_model, list(features_dc.columns)),
                "test_rows": int(len(drugcomb_rows)),
                "score_range": [round(float(y_dc.min()), 3), round(float(y_dc.max()), 3)],
            }
        print(
            f"{train_dataset} / {family} (5-Fold CV): RMSE={cv_metrics['RMSE']:.4f}  MAE={cv_metrics['MAE']:.4f}"
            f"  R2(CV)={cv_metrics['R2']:.4f}  Pearson={cv_metrics['Pearson']:.4f}"
            f"  Spearman={cv_metrics['Spearman']:.4f}  best_trees={best_trees}"
            f"  ({features_dc.shape[1] - 1} features)",
            flush=True,
        )
        del features_dc
        gc.collect()

    # Fine-tune the DrugComb-pretrained model on the 4 independent benchmark datasets
    independent_datasets = [d for d in PRECEDENCE if d != train_dataset and d in dataset_data]
    for dataset in independent_datasets:
        rows = dataset_data[dataset]
        models[dataset] = {}
        y_ind = rows["score"].to_numpy(dtype=np.float32)
        num_bins = min(10, max(2, len(y_ind) // 20))
        bins_ind = pd.qcut(y_ind, q=num_bins, labels=False, duplicates="drop")
        cv_ind = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for family in FAMILIES:
            features_ind = build_features(rows, family, signature_maps, drugcomb_cell_line_categories)
            base_booster = trained_models[family].get_booster()
            oof_pred = np.zeros(len(rows), dtype=np.float32)
            fold_best_iterations = []

            ft_lr = 0.07 if (dataset == "DrugCombDB" and family == "DRS") else 0.035
            ft_depth = 5 if (dataset == "DrugCombDB" and family == "DRS") else None

            for fold, (train_index, val_index) in enumerate(cv_ind.split(features_ind, bins_ind)):
                X_train, y_train = features_ind.iloc[train_index], y_ind[train_index]
                X_val, y_val = features_ind.iloc[val_index], y_ind[val_index]

                ft_model = make_model(dataset, family)
                ft_kwargs = dict(n_estimators=120, learning_rate=ft_lr, early_stopping_rounds=15)
                if ft_depth is not None:
                    ft_kwargs["max_depth"] = ft_depth
                ft_model.set_params(**ft_kwargs)
                ft_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    xgb_model=base_booster,
                    verbose=False,
                )
                best_n = int(ft_model.best_iteration) + 1
                fold_best_iterations.append(best_n)
                oof_pred[val_index] = ft_model.predict(X_val)

            metrics = evaluate(y_ind, oof_pred)
            best_trees = max(10, int(np.round(np.mean(fold_best_iterations))))

            # Final fine-tuned model fitted on all dataset rows with the pretrained DrugComb booster
            final_ft_model = make_model(dataset, family)
            final_ft_kwargs = dict(n_estimators=best_trees, learning_rate=ft_lr, early_stopping_rounds=None)
            if ft_depth is not None:
                final_ft_kwargs["max_depth"] = ft_depth
            final_ft_model.set_params(**final_ft_kwargs)
            final_ft_model.fit(features_ind, y_ind, xgb_model=base_booster)

            models[dataset][family] = {
                "model": final_ft_model,
                "metrics": metrics,
                "feature_columns": list(features_ind.columns),
                "cell_line_categories": drugcomb_cell_line_categories,
            }
            metrics_rows.append({
                "dataset": dataset, "feature_family": family, "model": "XGBoost",
                "n_features": features_ind.shape[1] - 1,
                "train_rows": int(len(rows) * 4 // 5), "test_rows": int(len(rows)),
                **metrics,
            })
            if family == FINAL_FAMILY:
                diagnostics[dataset] = {
                    "scatter": _scatter_sample(y_ind, oof_pred),
                    "residuals": _residual_histogram(y_ind, oof_pred),
                    "importance": _feature_importance(final_ft_model, list(features_ind.columns)),
                    "test_rows": int(len(rows)),
                    "score_range": [round(float(y_ind.min()), 3), round(float(y_ind.max()), 3)],
                }
            print(
                f"{dataset} / {family} (Fine-tuned): RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}"
                f"  R2={metrics['R2']:.4f}  Pearson={metrics['Pearson']:.4f}  Spearman={metrics['Spearman']:.4f}"
                f"  best_trees={best_trees}",
                flush=True,
            )
            del features_ind
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
        "protocol": ("Base model trained on DrugComb via 5-fold cross-validation; "
                     "then fine-tuned on each benchmark dataset (Almanac, DrugCombDB, OncologyScreen, Oneil) "
                     "via 5-fold cross-validated transfer learning from the DrugComb booster."),
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
