"""Runtime helpers for the author's drug-resistance signatures."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
SIGNATURE_MAP = ROOT / "datasets" / "signatures" / "DEGsenWithMASK.npy"
SIGNATURE_MAPS = {
    "DS": ROOT / "datasets" / "signatures" / "DEGWithMASK.npy",
    "DRS": SIGNATURE_MAP,
}
SIGNATURE_COLUMNS = (
    [f"DRS_drug1_{i:03d}" for i in range(1, 166)]
    + [f"DRS_drug2_{i:03d}" for i in range(1, 166)]
)


def canonical_smiles(value):
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


@lru_cache(maxsize=2)
def load_signature_map(family="DRS"):
    if family not in SIGNATURE_MAPS:
        raise ValueError(f"Unknown signature family: {family}")
    path = SIGNATURE_MAPS[family]
    if not path.exists():
        raise RuntimeError(f"{family} signature map is missing. Restore {path.relative_to(ROOT)}.")
    loaded = np.load(path, allow_pickle=True)
    source = loaded.item() if loaded.shape == () else loaded.flat[0]
    signatures = {}
    for smiles, components in source.items():
        canonical = canonical_smiles(smiles)
        if canonical is None or not isinstance(components, (tuple, list)) or len(components) != 2:
            continue
        if not isinstance(components, (tuple, list)) or len(components) != 2:
            continue
        vector = np.asarray(components[0], dtype=np.float64).ravel()
        mask = np.asarray(components[1], dtype=np.float64).ravel()
        if vector.shape != (165,) or mask.shape != (165,) or not np.isfinite(vector).all() or not np.isfinite(mask).all():
            continue
        if canonical in signatures and not np.array_equal(signatures[canonical], vector):
            raise ValueError(f"Conflicting DRS signatures for canonical SMILES {canonical}")
        signatures[canonical] = vector
    return signatures


def load_drs_signatures():
    return load_signature_map("DRS")


def ordered_drugs(drug1_smiles, drug2_smiles):
    canonical1, canonical2 = canonical_smiles(drug1_smiles), canonical_smiles(drug2_smiles)
    if canonical1 is None:
        raise ValueError("Drug 1 is not a valid SMILES structure.")
    if canonical2 is None:
        raise ValueError("Drug 2 is not a valid SMILES structure.")
    return tuple(sorted((canonical1, canonical2)))


def make_signature_features(drug1_smiles, drug2_smiles, cell_line, family):
    signatures = load_signature_map(family)
    canonical1, canonical2 = ordered_drugs(drug1_smiles, drug2_smiles)
    missing = [f"Drug {index}" for index, smiles in enumerate((canonical1, canonical2), start=1)
               if smiles not in signatures]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} has no DRS signature. This model supports only compounds in the "
            f"project's {len(signatures)}-drug {family} signature map."
        )
    values = np.concatenate([signatures[canonical1], signatures[canonical2]])
    columns = [f"{family}_drug1_{i:03d}" for i in range(1, 166)]
    columns += [f"{family}_drug2_{i:03d}" for i in range(1, 166)]
    record = {name: float(values[index]) for index, name in enumerate(columns)}
    record["cellline"] = str(cell_line)
    import pandas as pd
    return pd.DataFrame([record], columns=columns + ["cellline"])


def make_structure_features(drug1_smiles, drug2_smiles, cell_line):
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    values = []
    for smiles in ordered_drugs(drug1_smiles, drug2_smiles):
        molecule = Chem.MolFromSmiles(smiles)
        array = np.zeros(512, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), array)
        values.append(array)
    flat = np.concatenate(values)
    columns = [f"drug{drug}_bit_{bit}" for drug in (1, 2) for bit in range(512)]
    record = {name: int(flat[index]) for index, name in enumerate(columns)}
    record["cellline"] = str(cell_line)
    import pandas as pd
    return pd.DataFrame([record], columns=columns + ["cellline"])
