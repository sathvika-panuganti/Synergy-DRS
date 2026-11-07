# DrugComb and signature data provenance

The original DrugComb and DrugCombDB tables are distinct datasets. The project's previous DrugComb raw table was an exact copy of DrugCombDB. DrugComb has been restored from the author's repository's `dbs/DrugComb_LINCS_PRISM.csv`; DrugCombDB remains sourced from `dbs/DrugCombDBLINCS_PRISM.csv`.

- Author repository: https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS
- Author DrugComb source: https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/dbs/DrugComb_LINCS_PRISM.csv
- Author DrugCombDB source: https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/dbs/DrugCombDBLINCS_PRISM.csv
- DRS vector source: https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/data/DEGsenWithMASK.npy
- Author DRS feature preparation: https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/models/ML%20models/prepareLINCSsenDEG.py
- Author DS feature preparation: https://github.com/mozaffarilegha/DrugCombinationPrediction_DRS/blob/main/models/ML%20models/prepareLINCS.py

The legacy 330-column pair table retained rows when both drug SMILES were present in the author's DRS mapping and used the first 165-value signature component for each drug. This reproduces the original processed DrugCombDB DRS values exactly. The second 165-value component in each map is a mask, not another gene-expression feature; the comparison now excludes it and uses 330 pair-level signature columns (165 per drug) for both DS and DRS. The author scripts also append a 24,474-value cell-line graph vector; this project retains the cell-line identifier as a categorical feature instead, so the evaluation remains a lightweight comparison rather than an exact paper reproduction.

The existing DrugComb metric rows and saved notebook outputs were generated from the duplicate DrugCombDB data. They are historical only and must be retrained before reporting DrugComb performance on this corrected dataset.

| Dataset | Raw rows | Rows with both DRS mappings | Processed artifact |
|---|---:|---:|---|
| DrugComb | 63,468 | 48,766 | `DrugComb_LINCS_PRISM_final_without_last_score.zip` |
| DrugCombDB | 17,144 | 17,139 | `DrugCombDBLINCS_PRISM_final_without_last_score.zip` |

The earlier duplicate DrugComb CSV was preserved outside the repository at the task's temporary backup path while the project now uses the corrected compressed archive.
