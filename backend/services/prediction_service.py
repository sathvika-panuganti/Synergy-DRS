from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import json
from pathlib import Path

import joblib
from rdkit import Chem
from services.signature_features import (
    FINAL_FAMILY,
    make_family_frame,
    ordered_drugs,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = ROOT / "backend" / "models" / "combined_drs_structure_xgboost.joblib"

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
    """Load the signature bundle. The DRS model serves; there is no fallback."""
    if not MODEL_FILE.exists():
        raise RuntimeError(
            "The DRS model is not installed. Run "
            "`python backend/services/combined_feature_training.py` from the repository root, "
            "then restart Flask."
        )
    bundle = joblib.load(MODEL_FILE)
    if bundle.get("final_family") != FINAL_FAMILY:
        raise RuntimeError(
            f"{MODEL_FILE.name} was not produced by the current trainer "
            f"(final family is {bundle.get('final_family')!r}, expected {FINAL_FAMILY!r}). "
            "Retrain before serving predictions."
        )
    return bundle


def _select_dataset(bundle, pair1, pair2, cell_line):
    """Pick the source dataset whose model covers this pair and cell line.

    The dataset is an internal routing decision only: every dataset is served by
    its own DRS model, and the dataset name is not part of the reported score.
    """
    observed = bundle["observed_scores"].get((pair1, pair2, cell_line))
    if observed:
        return observed["dataset"], observed
    dataset = next((name for name in bundle["precedence"]
                    if pair1 in bundle["domains"][name]["drugs"]
                    and pair2 in bundle["domains"][name]["drugs"]
                    and cell_line in bundle["domains"][name]["cell_lines"]), None)
    if dataset is not None:
        return dataset, None

    # Say which part of the input is unsupported rather than failing generically.
    supported_drugs = set(bundle["supported_drugs"])
    unsupported = [label for label, smiles in (("Drug 1", pair1), ("Drug 2", pair2))
                   if smiles not in supported_drugs]
    if unsupported:
        raise ValueError(
            f"{' and '.join(unsupported)} has no drug resistance signature in this project. "
            f"The model covers the {len(supported_drugs)} compounds shared by LINCS and GDSC; "
            "pick one of the suggested drugs."
        )
    if cell_line not in set(bundle["supported_cell_lines"]):
        raise ValueError(f"'{cell_line}' is not one of the project's trained cell lines.")
    raise ValueError(
        "Both compounds are supported, but no source dataset covers this pair together with "
        f"cell line {cell_line}. Try another cell line, or pick a recorded example."
    )


def predict(drug1_input, drug2_input, cell_line):
    original_drug1, original_drug2 = drug1_input.strip(), drug2_input.strip()
    drug1_smiles, drug1_resolved = _resolve_structure(original_drug1, "Drug 1")
    drug2_smiles, drug2_resolved = _resolve_structure(original_drug2, "Drug 2")
    bundle = _load_model()
    pair1, pair2 = ordered_drugs(drug1_smiles, drug2_smiles)
    cell_line = cell_line.strip()
    dataset, observed = _select_dataset(bundle, pair1, pair2, cell_line)

    entry = bundle["models"][dataset][FINAL_FAMILY]
    # Raises ValueError when a DRS signature is missing, so the API reports the
    # gap instead of scoring an incomplete feature vector.
    features = make_family_frame(FINAL_FAMILY, pair1, pair2, cell_line,
                                 entry["cell_line_categories"])
    expected = list(entry["feature_columns"])
    if list(features.columns) != expected:
        raise RuntimeError(
            "DRS feature layout does not match the trained model "
            f"({len(features.columns)} built vs {len(expected)} expected). Retrain the model."
        )
    score = float(entry["model"].predict(features)[0])

    # Every feature family is scored for the same input so the UI can compare
    # them side by side; only the DRS model's score is the final answer.
    family_scores = []
    for family in ("Structure", "DS", "DRS"):
        model_entry = bundle["models"][dataset].get(family)
        if not model_entry:
            continue
        if family == FINAL_FAMILY:
            value = score
        else:
            frame = make_family_frame(family, pair1, pair2, cell_line,
                                      model_entry["cell_line_categories"])
            value = float(model_entry["model"].predict(frame)[0])
        metrics = model_entry["metrics"]
        family_scores.append({
            "family": family,
            "score": round(value, 4),
            "final": family == FINAL_FAMILY,
            "n_features": len(model_entry["feature_columns"]) - 1,
            "rmse": round(float(metrics["RMSE"]), 4),
            "mae": round(float(metrics["MAE"]), 4),
            "r2": round(float(metrics["R2"]), 4),
        })

    return {
        "score": round(score, 4),
        "classification": "SYNERGISTIC REACTION" if score > 10 else "NOT A SYNERGISTIC REACTION",
        "model": "XGBoost",
        "feature_family": FINAL_FAMILY,
        "selected_model": f"XGBoost · {FINAL_FAMILY}",
        "n_features": len(expected) - 1,
        "family_scores": family_scores,
        "drug1_input": original_drug1,
        "drug2_input": original_drug2,
        "drug1_smiles": drug1_smiles,
        "drug2_smiles": drug2_smiles,
        "drug1_resolved": drug1_resolved,
        "drug2_resolved": drug2_resolved,
        "cell_line": cell_line,
    }
