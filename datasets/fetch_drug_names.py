"""Build a SMILES -> common drug name map for the UI labels.

One-off helper: it queries PubChem for the preferred title of every compound in
the trained model bundle and writes datasets/metadata/drug_names.json. The map
is display-only; predictions never depend on it, and the app works without it.

    python datasets/fetch_drug_names.py
"""
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import joblib

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "backend" / "models" / "combined_drs_structure_xgboost.joblib"
OUTPUT = ROOT / "datasets" / "metadata" / "drug_names.json"
ENDPOINT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/property/Title/JSON"


def pubchem_title(smiles):
    request = Request(ENDPOINT.format(quote(smiles, safe="")),
                      headers={"User-Agent": "Synergy-DRS/1.0"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    properties = payload.get("PropertyTable", {}).get("Properties", [])
    title = properties[0].get("Title") if properties else None
    if not title or len(title) > 40:
        return None
    return title


def main():
    if not BUNDLE.exists():
        raise SystemExit(f"Train the combined model first: {BUNDLE} is missing.")
    drugs = joblib.load(BUNDLE)["supported_drugs"]
    existing = {}
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8") as stream:
            existing = json.load(stream)

    resolved, failed = dict(existing), []
    for index, smiles in enumerate(drugs, start=1):
        if smiles in resolved:
            continue
        title = pubchem_title(smiles)
        if title:
            resolved[smiles] = title
        else:
            failed.append(smiles)
        print(f"[{index}/{len(drugs)}] {title or 'unresolved'}", flush=True)
        time.sleep(0.25)  # stay inside PubChem's request-rate guidance

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(resolved, stream, indent=2, sort_keys=True)
    print(f"\nSaved {len(resolved)} names to {OUTPUT}; {len(failed)} unresolved.")


if __name__ == "__main__":
    main()
