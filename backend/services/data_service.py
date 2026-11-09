"""Options, recorded combination examples, and the evaluation report for the UI."""
from pathlib import Path
from functools import lru_cache
import json

import joblib

ROOT = Path(__file__).resolve().parents[2]
PREDICTION_MODEL = ROOT / "backend" / "models" / "combined_drs_structure_xgboost.joblib"
DRUG_NAMES_FILE = ROOT / "datasets" / "metadata" / "drug_names.json"
METRICS_FILE = ROOT / "frontend" / "data" / "metrics.json"
REPORT_FILE = ROOT / "frontend" / "data" / "model_report.json"
OPTIONS_FALLBACK = ROOT / "frontend" / "data" / "options.json"

SYNERGY_THRESHOLD = 10.0
EXAMPLES_PER_GROUP = 24
SUGGESTION_LIMIT = 60


@lru_cache(maxsize=1)
def _prediction_bundle():
    if not PREDICTION_MODEL.exists():
        return None
    return joblib.load(PREDICTION_MODEL)


@lru_cache(maxsize=1)
def _drug_names():
    """Optional SMILES -> common drug name map, used only for display labels."""
    if not DRUG_NAMES_FILE.exists():
        return {}
    try:
        with DRUG_NAMES_FILE.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): str(value) for key, value in data.items() if value}


def _label(smiles):
    return _drug_names().get(smiles, "")


@lru_cache(maxsize=1)
def get_options():
    """Cell lines, drug suggestions, and recorded examples pooled over all datasets."""
    bundle = _prediction_bundle()
    if bundle is None:
        with OPTIONS_FALLBACK.open(encoding="utf-8") as stream:
            return json.load(stream)

    names = _drug_names()
    supported_cells = set(bundle.get("supported_cell_lines", []))
    rows, best_for_drug = [], {}
    # observed_scores is already pooled across Almanac, DrugComb, DrugCombDB,
    # OncologyScreen and Oneil and keyed by canonical unordered pair + cell line,
    # so reversed pairs and cross-dataset duplicates collapse to one entry.
    for (drug1, drug2, cell_line), observed in bundle.get("observed_scores", {}).items():
        if not drug1 or not drug2 or not cell_line or cell_line not in supported_cells:
            continue
        score = float(observed["score"])
        rows.append({
            "drug1": drug1, "drug2": drug2, "cell_line": cell_line, "score": score,
            "drug1_label": names.get(drug1, ""), "drug2_label": names.get(drug2, ""),
        })
        for drug in (drug1, drug2):
            if drug not in best_for_drug or score > best_for_drug[drug]["score"]:
                best_for_drug[drug] = {"name": drug, "label": names.get(drug, ""), "score": score}

    synergistic = sorted((row for row in rows if row["score"] > SYNERGY_THRESHOLD),
                         key=lambda row: row["score"], reverse=True)[:EXAMPLES_PER_GROUP]
    not_synergistic = sorted((row for row in rows if row["score"] <= SYNERGY_THRESHOLD),
                             key=lambda row: row["score"])[:EXAMPLES_PER_GROUP]

    # Recorded scores decide which group an example belongs to, but they are not
    # published: the UI shows only the combination, so nothing competes with the
    # model's own predicted score.
    def example(row, classification):
        return {key: value for key, value in row.items() if key != "score"} | {
            "classification": classification}

    suggestions = [{"name": item["name"], "label": item["label"]}
                   for item in sorted(best_for_drug.values(), key=lambda item: item["score"],
                                      reverse=True)[:SUGGESTION_LIMIT]]
    return {
        "cell_lines": sorted(supported_cells),
        "suggestions": suggestions,
        "recommendations": {
            "synergistic": [example(row, "SYNERGISTIC") for row in synergistic],
            "not_synergistic": [example(row, "NOT SYNERGISTIC") for row in not_synergistic],
        },
        "example_pool_size": len(rows),
        "final_model": bundle.get("final_family", ""),
    }


def _read_json(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def get_dashboard_data():
    """Recorded evaluation: the legacy model comparison plus the signature-model run."""
    dashboard = _read_json(METRICS_FILE) or {"evaluation": {}, "metrics": []}
    dashboard["report"] = _read_json(REPORT_FILE)
    return dashboard
