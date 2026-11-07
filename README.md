# Synergy-DRS

Flask and vanilla JavaScript research dashboard for drug combination synergy datasets.

## Run locally

From the repository root, create and activate a virtual environment, then install the backend requirements:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python datasets\prepare_signature_features.py
python -m backend.services.multi_dataset_xgboost_training
python backend\app.py
```

Open http://127.0.0.1:5000. The same Flask app serves the frontend and `/api/options`, `/api/dashboard`, and `/api/predict`. Model training is a one-time setup step; the generated model is saved in `backend/models/` and is loaded once by the API.

## Deploy on Render

The repository includes `render.yaml` and a Gunicorn WSGI entry point. Connect the repository in Render and deploy the `synergy-drs` web service. Its build command installs the requirements, rebuilds the DS/DRS feature archives from the raw project datasets and signature maps, trains the XGBoost prediction bundle, and regenerates the 60-row dashboard report. It starts the Flask app with one worker.

Keep the five raw CSV files under `datasets/raw/` and both author signature maps under `datasets/signatures/` in the deployed source. The raw CSVs are tracked; the two signature maps must be included in the Git commit because the build uses them. The feature archives, trained model, and dashboard report are generated during the build. Render must allow outbound HTTPS for optional PubChem drug-name lookups; SMILES input does not need that lookup.

The prediction endpoint is a research prototype. It only supports compounds with author-provided signatures and cell lines in the selected dataset. The dashboard uses a single grouped holdout, as requested, rather than the paper's repeated cross-validation protocol; scores should be presented as project holdout estimates, not direct reproductions of the paper's reported estimates. SynergyX is excluded from the current scope.

## Current project data status

After the multi-dataset model bundle is trained, the API filters cell-line options, compound suggestions, and examples to the supported domain across the five datasets. The dashboard compares LASSO, Random Forest, AdaBoost, and XGBoost across Structure, DS, and DRS features on all five datasets, and reports MSE, RMSE, MAE, R², explained variance, Pearson, Spearman, and AUC. Legacy `.xls` metric files are not displayed there.

The prediction API uses XGBoost models for Structure, DS, and DRS features, trained separately on each of the five datasets. It evaluates each family with one grouped 80/20 holdout split per dataset, then refits on all rows. The final prediction uses the family with the lowest holdout RMSE for the selected dataset. For an already-recorded drug pair and cell line, the observed-score source follows `DrugComb > Almanac > DrugCombDB > OncologyScreen > Oneil`; ties within a source are averaged because the feature archives do not retain dose columns.

After installing dependencies, prepare the archives and train or rebuild the model bundle:

```powershell
pip install -r backend\requirements.txt
python datasets\prepare_signature_features.py
python -m backend.services.multi_dataset_xgboost_training
```

The runtime maps each input structure to the author's DS and DRS signature maps and also computes Morgan fingerprints. It returns all three XGBoost scores, the selected final score, and any precedence-resolved observed score and source. Compounds must be present in the author-provided signature maps, and the selected dataset must contain the requested cell line. Generated model artifacts are Git-ignored and recreated locally.

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
