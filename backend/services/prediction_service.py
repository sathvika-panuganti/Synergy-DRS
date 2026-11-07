from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import json
from pathlib import Path

import joblib
from rdkit import Chem
from services.signature_features import (
    make_signature_features,
    make_structure_features,
    ordered_drugs,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "backend" / "models" / "xgboost_precedence_ensemble.joblib"

PUBCHEM_TIMEOUT_SECONDS = 8


@lru_cache(maxsize=512)
def _lookup_pubchem_smiles(drug_name):
    """Resolve a drug name to a PubChem SMILES string."""
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{quote(drug_name, safe='')}/property/SMILES/JSON")
    request = Request(url, headers={"User-Agent": "Synergy-DRS/1.0"})
    try:
        with urlopen(request, timeout=PUBCHEM_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code == 404:
            raise ValueError(f"Could not find '{drug_name}' in PubChem. Check the name or enter its SMILES.") from error
        raise RuntimeError("PubChem name lookup is temporarily unavailable. Try again or enter SMILES directly.") from error
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError("PubChem name lookup is temporarily unavailable. Check your connection or enter SMILES directly.") from error

    properties = payload.get("PropertyTable", {}).get("Properties", [])
    if len(properties) > 1:
        raise ValueError(f"'{drug_name}' matches multiple PubChem compounds. Use a more specific name or enter SMILES.")
    smiles = properties[0].get("SMILES") if properties else None
    if not smiles:
        raise ValueError(f"PubChem did not return a usable structure for '{drug_name}'. Enter its SMILES instead.")
    return smiles


def _resolve_structure(value, label):
    value = value.strip()
    if Chem.MolFromSmiles(value) is not None:
        return value, False
    try:
        smiles = _lookup_pubchem_smiles(value)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"PubChem returned an invalid structure for {label.lower()}. Enter SMILES directly.")
    return smiles, True


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_FILE.exists():
        raise RuntimeError(
            "The multi-dataset XGBoost model is not installed. Run `python -m backend.services.multi_dataset_xgboost_training` "
            "from the repository root, then restart Flask."
        )
    return joblib.load(MODEL_FILE)


def predict(drug1_input, drug2_input, cell_line):
    original_drug1, original_drug2 = drug1_input.strip(), drug2_input.strip()
    drug1_smiles, drug1_resolved = _resolve_structure(original_drug1, "Drug 1")
    drug2_smiles, drug2_resolved = _resolve_structure(original_drug2, "Drug 2")
    bundle = _load_model()
    pair1, pair2 = ordered_drugs(drug1_smiles, drug2_smiles)
    cell_line = cell_line.strip()
    observed = bundle["observed_scores"].get((pair1, pair2, cell_line))
    if observed:
        dataset = observed["dataset"]
    else:
        dataset = next((name for name in bundle["precedence"]
                        if pair1 in bundle["domains"][name]["drugs"]
                        and pair2 in bundle["domains"][name]["drugs"]
                        and cell_line in bundle["domains"][name]["cell_lines"]), None)
        if dataset is None:
            raise ValueError("No project dataset contains both compounds and this cell line in its supported domain.")

    feature_builders = {
        "Structure": lambda: make_structure_features(pair1, pair2, cell_line),
        "DS": lambda: make_signature_features(pair1, pair2, cell_line, "DS"),
        "DRS": lambda: make_signature_features(pair1, pair2, cell_line, "DRS"),
    }
    scores = {}
    for family, build_features in feature_builders.items():
        model_info = bundle["models"][dataset][family]
        scores[family] = float(model_info["model"].predict(build_features())[0])
    selected_family = bundle["best_feature_family"][dataset]
    score = scores[selected_family]
    score_sources = ([
        {"dataset": observed["dataset"], "score": round(float(observed["score"]), 4)}
    ] + [
        {"dataset": item["dataset"], "score": round(float(item["score"]), 4)}
        for item in observed.get("other_sources", [])
    ]) if observed else []
    return {
        "score": round(score, 4),
        "classification": "SYNERGISTIC REACTION" if score > 10 else "NOT A SYNERGISTIC REACTION",
        "scores": {family: round(value, 4) for family, value in scores.items()},
        "selected_model": f"XGBoost + {selected_family}",
        "dataset_selection": "observed-score precedence" if observed else "first supported dataset by precedence",
        "observed_score": round(float(observed["score"]), 4) if observed else None,
        "observed_dataset": observed["dataset"] if observed else None,
        "score_sources": score_sources,
        "drug1_input": original_drug1,
        "drug2_input": original_drug2,
        "drug1_smiles": drug1_smiles,
        "drug2_smiles": drug2_smiles,
        "drug1_resolved": drug1_resolved,
        "drug2_resolved": drug2_resolved,
        "cell_line": cell_line,
        "model": "XGBoost",
        "dataset": dataset,
        "selected_feature_family": selected_family,
    }
