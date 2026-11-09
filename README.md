# Synergy-DRS

Flask and vanilla JavaScript research dashboard for drug combination synergy datasets.

The final predicted score comes from a single XGBoost model that receives **Drug Resistance
Signature (DRS) features and molecular Structure features together** in one feature vector.

## Run locally (macOS / Linux)

From the repository root:

```bash
python3 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt

python datasets/prepare_signature_features.py          # build DS and DRS pair archives
python backend/services/combined_feature_training.py   # train the DRS model and baselines
python datasets/fetch_drug_names.py                    # optional: SMILES -> drug-name labels

cd backend && python app.py
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` and backslash paths.

Open http://127.0.0.1:5000. The same Flask app serves the frontend and `/api/options`, `/api/dashboard`, and `/api/predict`. Model training is a one-time setup step; the generated model is saved in `backend/models/` and is loaded once by the API.

## Deploy on Render

The repository includes `render.yaml` and a Gunicorn WSGI entry point. Connect the repository in Render and deploy the `synergy-drs` web service. Its build command installs the requirements, rebuilds the DS/DRS feature archives from the raw project datasets and signature maps, trains the XGBoost prediction bundle, and regenerates the 60-row dashboard report. It starts the Flask app with one worker.

Keep the five raw CSV files under `datasets/raw/` and both author signature maps under `datasets/signatures/` in the deployed source. The raw CSVs are tracked; the two signature maps must be included in the Git commit because the build uses them. The feature archives, trained model, and dashboard report are generated during the build. Render must allow outbound HTTPS for optional PubChem drug-name lookups; SMILES input does not need that lookup.

The prediction endpoint is a research prototype. It only supports compounds with author-provided signatures and cell lines in the selected dataset. The dashboard uses a single grouped holdout, as requested, rather than the paper's repeated cross-validation protocol; scores should be presented as project holdout estimates, not direct reproductions of the paper's reported estimates. SynergyX is excluded from the current scope.

## Current project data status

The API filters cell-line options, compound suggestions, and examples to the supported domain
across the five datasets. The "Try a combination" list pools recorded examples from all five
sources into one deduplicated set (reversed drug pairs collapse to one entry) and carries no
dataset label, and no recorded experimental score — a prediction result shows the model's own
scores only. Each prediction returns all three feature-family scores for the same input so they can
be compared side by side, with DRS marked as the final model.

The dashboard table reports MSE, RMSE, MAE, R², Pearson and Spearman. AUC and explained variance
are not shown: this is a regression task, and the reported numbers are the regression metrics.
Every XGBoost row — Structure, DS and DRS — comes from the signature-model run so
the three families are directly comparable; the LASSO, Random Forest and AdaBoost rows are the
earlier comparison run. Five research-style figures are drawn from the saved evaluation data:
feature-family comparison, actual vs predicted with the ideal line, residual distribution,
gain-based feature importance split into the drug-1 and drug-2 signature blocks, and a
cross-dataset R² comparison. Legacy `.xls` metric files are not displayed.

### Final prediction model

`backend/services/combined_feature_training.py` trains, for each of the five datasets:

| Feature family | Inputs | Role |
| --- | --- | --- |
| Structure | 1024 Morgan bits + 12 descriptors (MolWt, LogP, HBD, HBA, TPSA, rotatable bonds, per drug) | baseline |
| DS | 330 drug-signature values | baseline |
| **DRS** | **330 drug-resistance-signature values** | **served model** |

The DRS family is the method described in the research paper, so it is what the API serves;
Structure and DS are kept only as baselines for comparison. All three use identical
hyper-parameters and the same grouped 80/20 holdout split by canonical unordered drug pair, so no
pair appears in both training and test. Metrics come only from the held-out pairs; the served model
is then refit on every row of its dataset. The synergy score is never an input feature, and the API
never falls back to another feature family.

At prediction time the same feature builders in `backend/services/signature_features.py` are used,
and the built column order is compared against the trained model's column list before scoring. If
a compound has no DRS signature, or the pair and cell line fall outside the trained domain, the
API returns a 400 with an explanation instead of an unreliable score.

The dataset whose model answers a request is chosen internally by coverage and the precedence
`DrugComb > Almanac > DrugCombDB > OncologyScreen > Oneil`; repeated measurements of a pair and
cell line within a source are averaged because the feature archives do not retain dose columns.
The dataset name is not part of the reported score and is not shown in the UI.

Rebuild the archives and retrain the served model with:

```bash
python datasets/prepare_signature_features.py
python backend/services/combined_feature_training.py
```

That writes `backend/models/combined_drs_structure_xgboost.joblib`,
`results/combined_model_metrics.csv`, `results/combined_model_diagnostics.json` and
`frontend/data/model_report.json` (the evaluation data behind the dashboard figures).
Generated model artifacts are Git-ignored and recreated locally.

`backend/services/multi_dataset_xgboost_training.py` and its
`xgboost_precedence_ensemble.joblib` bundle are kept for reference; they are the earlier
single-family pipeline and are no longer used for serving.

## Train the research notebooks

The notebooks in `datasets/trained` read their input tables from `datasets/processed`. Launch Jupyter from the repository root so that outputs and relative paths stay predictable:

```powershell
pip install -r backend\requirements.txt
pip install jupyter xgboost matplotlib seaborn
jupyter notebook
```

Open the required notebook under `datasets/trained`. The notebooks locate the repository root automatically when started from the root or a nested project folder.

## Build DS and DRS signature features

The author-provided signature maps are stored under `datasets/signatures`. From the repository root, run:

```powershell
python datasets\prepare_signature_features.py
```

This creates separate compressed DS and DRS pair-feature archives under `datasets/processed/signature_features` for the five raw datasets. Each drug contributes the map's 165-value signature; the separate 165-value mask is not concatenated as biological features, so each pair has 330 signature features total, plus the cell-line identifier and synergy score. Rows without signatures for both drugs are excluded; each archive records its source hashes and row counts in `metadata.json`. The project encodes cell line categorically; the author's preparation scripts also use a 24,474-value cell-line graph vector, so this is not an exact reproduction of that feature pipeline. This step extracts features only; it does not retrain models or update existing metrics. The generated archives and processed CSV/XLS intermediates are Git-ignored and can be rebuilt locally, so they do not need to be uploaded with the source code.

## Compare Structure, DS, and DRS models

Install the model-training dependencies and run the deployment dashboard evaluation (one grouped holdout only; SynergyX is intentionally excluded):

```powershell
pip install -r datasets\requirements-training.txt
python datasets\compare_classical_holdout.py
```

The comparison uses the same canonical unordered drug-pair grouping and 80/20 split seed for every feature family and model within each dataset. It writes the detailed CSV to `results/classical_grouped_holdout_summary.csv` and refreshes `frontend/data/metrics.json`. For a practical single-run benchmark, LASSO uses alpha 0.01, 5,000 iterations and 0.01 tolerance; Random Forest uses 50 trees, depth 20, 25% feature sampling and leaf size 2; AdaBoost uses 25 depth-4 trees at learning rate 0.05; XGBoost uses 300 depth-4 trees at learning rate 0.05. Results are single-holdout estimates; they do not include repeated cross-validation confidence intervals. The deployed prediction API continues to use the separate multi-dataset XGBoost models described above.
