from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "Almanac": "AlmanacLINCS_PRISM.csv",
    "DrugComb": "DrugComb_LINCS_PRISM.csv",
    "DrugCombDB": "DrugCombDBLINCS_PRISM.csv",
    "OncologyScreen": "OncologyScreenLINCS_PRISM.csv",
    "Oneil": "OneilLINCS_PRISM.csv",
}
MODEL_FILES = {"AdaBoost": "adaboost_separate_dataset_metrics.xls", "Lasso": "lasso_separate_dataset_metrics.xls", "Random Forest": "random_forest_separate_dataset_metrics.xls", "XGBoost": "xgboost_separate_dataset_metrics.xls"}
PREDICTION_MODEL = ROOT / "backend" / "models" / "xgboost_precedence_ensemble.joblib"

@lru_cache(maxsize=1)
def _prediction_bundle():
    if not PREDICTION_MODEL.exists():
        return None
    return joblib.load(PREDICTION_MODEL)


def _prediction_domain():
    bundle = _prediction_bundle()
    if bundle is None:
        return None
    return set(bundle.get("supported_drugs", [])), set(bundle.get("supported_cell_lines", []))

def _read_dataset(path):
    if path.suffix.lower() in (".xls", ".xlsx"):
        try:
            return pd.read_excel(path)
        except Exception:
            # Some project files use an .xls suffix but contain CSV text.
            return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def get_options():
    options_path = ROOT / "frontend" / "data" / "options.json"
    bundle = _prediction_bundle()
    if bundle is None:
        with options_path.open(encoding="utf-8") as file:
            return json.load(file)

    cells = bundle.get("supported_cell_lines", [])
    suggestions = {}
    scored_rows = []
    for (drug1, drug2, cell_line), observed in bundle.get("observed_scores", {}).items():
        score = float(observed["score"])
        scored_rows.append({
            "drug1": drug1, "drug2": drug2, "cell_line": cell_line,
            "score": score, "dataset": observed["dataset"],
        })
        for drug in (drug1, drug2):
            if drug not in suggestions or score > suggestions[drug]["score"]:
                suggestions[drug] = {"name": drug, "score": score, "dataset": observed["dataset"]}
    synergistic = sorted((row for row in scored_rows if row["score"] > 10), key=lambda row: row["score"], reverse=True)[:3]
    not_synergistic = sorted((row for row in scored_rows if row["score"] <= 10), key=lambda row: row["score"])[:3]
    recommendations = {
        "synergistic": [{**row, "classification": "SYNERGISTIC"} for row in synergistic],
        "not_synergistic": [{**row, "classification": "NOT SYNERGISTIC"} for row in not_synergistic],
    }
    return {
        "cell_lines": sorted(cells),
        "suggestions": sorted(suggestions.values(), key=lambda item: item["score"], reverse=True)[:30],
        "recommendations": recommendations,
    }

def get_dashboard_data():
    path = ROOT / "frontend" / "data" / "metrics.json"
    with path.open(encoding="utf-8") as file:
        return json.load(file)

def _num(value):
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None
