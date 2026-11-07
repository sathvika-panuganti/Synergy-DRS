"""Train the deployable baseline model with the notebook's input features."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "datasets" / "raw" / "OneilLINCS_PRISM.csv"
MODEL_FILE = ROOT / "backend" / "models" / "oneil_random_forest.joblib"
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
BITS = [f"drug{drug}_bit_{bit}" for drug in (1, 2) for bit in range(512)]

def fingerprint(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES in training data: {smiles}")
    array = np.zeros(512, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(GENERATOR.GetFingerprint(molecule), array)
    return array

def make_features(drug1, drug2, cell_line):
    values = np.concatenate([fingerprint(drug1), fingerprint(drug2)])
    record = {column: int(values[i]) for i, column in enumerate(BITS)}
    record["cellline"] = str(cell_line)
    return pd.DataFrame([record], columns=BITS + ["cellline"])

def train_and_save():
    data = pd.read_csv(DATA_FILE, low_memory=False)
    target_column = "score" if "score" in data.columns else "synergy_score"
    required = ["drugname1", "drugname2", "cellline", target_column]
    data = data[required].dropna().drop_duplicates().copy()
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
    data = data.dropna(subset=[target_column])

    fingerprints = []
    valid_rows = []
    for row_index, row in data.iterrows():
        try:
            fingerprints.append(np.concatenate([fingerprint(row.drugname1), fingerprint(row.drugname2)]))
            valid_rows.append(row_index)
        except ValueError:
            continue
    data = data.loc[valid_rows].reset_index(drop=True)
    matrix = np.asarray(fingerprints, dtype=np.uint8)
    features = pd.DataFrame(matrix, columns=BITS)
    features["cellline"] = data["cellline"].astype(str).values
    target = data[target_column].astype(float)

    fingerprint_prep = SimpleImputer(strategy="constant", fill_value=np.uint8(0))
    cell_line_prep = OneHotEncoder(handle_unknown="ignore")
    preprocess = ColumnTransformer([
        ("fingerprints", fingerprint_prep, BITS),
        ("cellline", cell_line_prep, ["cellline"]),
    ])
    model = make_pipeline(preprocess, RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "dataset": "Oneil", "model": "Random Forest baseline", "rows": len(data),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "mae": float(mean_absolute_error(y_test, predicted)),
        "r2": float(r2_score(y_test, predicted)),
    }
    # Fit the deployment model on all available rows after measuring held-out performance.
    model.fit(features, target)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics, "feature_columns": BITS + ["cellline"]}, MODEL_FILE)
    print(f"Saved model: {MODEL_FILE}")
    print(f"Held-out metrics: {metrics}")

if __name__ == "__main__":
    train_and_save()
