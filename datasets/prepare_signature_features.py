import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ROOT = next((p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
             if (p / "datasets" / "raw").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Could not locate the repository root containing datasets/raw.")

RAW_DIR = ROOT / "datasets" / "raw"
SIGNATURE_DIR = ROOT / "datasets" / "signatures"
OUTPUT_DIR = ROOT / "datasets" / "processed" / "signature_features"
MAPS = {
    "DS": SIGNATURE_DIR / "DEGWithMASK.npy",
    "DRS": SIGNATURE_DIR / "DEGsenWithMASK.npy",
}
DATASETS = {
    "Almanac": "AlmanacLINCS_PRISM.csv",
    "DrugComb": "DrugComb_LINCS_PRISM.csv",
    "DrugCombDB": "DrugCombDBLINCS_PRISM.csv",
    "OncologyScreen": "OncologyScreenLINCS_PRISM.csv",
    "Oneil": "OneilLINCS_PRISM.csv",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_signature_map(path):
    """Load the author's 165-value drug signature; the second array is a mask."""
    values = np.load(path, allow_pickle=True)
    mapping = values.item() if values.shape == () else values.flat[0]
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected a dictionary in {path}")

    result = {}
    for drug, components in mapping.items():
        if not isinstance(components, (tuple, list)) or len(components) != 2:
            raise ValueError(f"Unexpected signature components for {drug!r}")
        vector = np.asarray(components[0], dtype=np.float64).ravel()
        mask = np.asarray(components[1], dtype=np.float64).ravel()
        if vector.shape != (165,) or mask.shape != (165,) or not np.isfinite(vector).all() or not np.isfinite(mask).all():
            raise ValueError(f"Expected 165 signature values and a 165-value mask for {drug!r}")
        result[str(drug)] = vector
    return result


def make_pair_matrix(data, signature_map):
    """Return the matched subset and 330 signature features in drug1-then-drug2 order."""
    required = {"drugname1", "drugname2", "cellline"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")
    target = "score" if "score" in data.columns else "synergy_score"
    if target not in data.columns:
        raise ValueError("No score or synergy_score target column was found.")

    valid = data["drugname1"].isin(signature_map) & data["drugname2"].isin(signature_map)
    matched = data.loc[valid].copy().reset_index(drop=True)
    matrix = np.empty((len(matched), 330), dtype=np.float64)
    for row_index, (drug1, drug2) in enumerate(zip(matched.drugname1, matched.drugname2)):
        matrix[row_index, :165] = signature_map[drug1]
        matrix[row_index, 165:] = signature_map[drug2]
    matched["synergy_score"] = pd.to_numeric(matched[target], errors="coerce")
    matched = matched.drop(columns=[c for c in ("score",) if c in matched.columns])
    return matched, matrix, int((~valid).sum())


def write_archive(dataset_name, family, source_path, data, matched, matrix, dropped):
    feature_columns = (
        [f"{family}_drug1_{i:03d}" for i in range(1, 166)]
        + [f"{family}_drug2_{i:03d}" for i in range(1, 166)]
    )
    output = OUTPUT_DIR / f"{dataset_name}_{family}_features.zip"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"{dataset_name}_{family}_features.csv"
    metadata = {
        "dataset": dataset_name,
        "signature_family": family,
        "source_rows": int(len(data)),
        "rows_with_both_drug_signatures": int(len(matched)),
        "rows_dropped_missing_signature": dropped,
        "feature_columns": len(feature_columns),
        "signature_values_per_drug": 165,
        "feature_order": "drug1 165-value signature, then drug2 165-value signature; the second map array is a mask and is not treated as a feature",
        "target_column": "synergy_score",
        "source_file": source_path.name,
        "source_sha256": file_sha256(source_path),
        "signature_map_sha256": file_sha256(MAPS[family]),
        "cell_line_representation": "categorical cellline identifier; no author cell-graph vector included",
    }
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        with archive.open(csv_name, "w") as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
                frame = pd.concat(
                    [pd.DataFrame(matrix, columns=feature_columns), matched.reset_index(drop=True)],
                    axis=1,
                )
                frame.to_csv(text, index=False, float_format="%.17g")
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
    print(f"{dataset_name} {family}: {len(data)} rows -> {len(matched)} rows; saved {output}")


def prepare(dataset_names):
    signature_maps = {family: load_signature_map(path) for family, path in MAPS.items()}
    if set(signature_maps["DS"]) != set(signature_maps["DRS"]):
        raise ValueError("The DS and DRS signature maps do not contain the same drug keys.")
    for dataset_name in dataset_names:
        source_path = RAW_DIR / DATASETS[dataset_name]
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        data = pd.read_csv(source_path, low_memory=False)
        for family, mapping in signature_maps.items():
            matched, matrix, dropped = make_pair_matrix(data, mapping)
            write_archive(dataset_name, family, source_path, data, matched, matrix, dropped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build author-derived DS and DRS pair feature archives.")
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    prepare(args.datasets)
