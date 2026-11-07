# Dataset feature extraction

`../prepare_signature_features.py` builds separate pair-level DS and DRS tables from the author-provided signature maps. Each drug contributes the 165-value signature array; the separate 165-value mask is not appended as another feature. The pair input concatenates drug 1 and drug 2 (330 signature columns total), and retains `cellline` as a categorical field plus the regression target.

The author source scripts are `prepareLINCS.py` (DS) and `prepareLINCSsenDEG.py` (DRS):

- https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/models/ML%20models/prepareLINCS.py
- https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/models/ML%20models/prepareLINCSsenDEG.py

Signature maps are sourced from:

- https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/data/DEGWithMASK.npy
- https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/data/DEGsenWithMASK.npy

Rows lacking either drug signature are omitted and counted in each archive's metadata. This extractor retains the project's cell-line identifier. The author's preparation scripts additionally concatenate a 24,474-value cell-line graph vector; that large cell representation is not included in these archives, so the later model comparison should use the same cell-line encoding across Structure, DS, and DRS.
