# Synergy-DRS – Project Guide

## 1. Project Overview

Synergy-DRS focuses on drug combination synergy prediction using machine learning techniques.

This guide explains the repository structure, datasets, feature engineering process, model implementation, evaluation metrics and execution workflow.

## 2. Repository Organization

### backend/
Contains backend-related project components.

### datasets/
Contains the datasets used for drug combination synergy prediction.

The project works with the following five datasets:

- DrugComb
- DrugCombDB
- O'Neil
- ALMANAC
- OncologyScreen

### frontend/
Contains frontend-related project components and user interface resources.

### metadata/
Contains Jupyter notebooks implementing the machine learning models.

| Notebook | Algorithm |
|---|---|
| AdaBoost.ipynb | AdaBoost |
| Lasso.ipynb | Lasso Regression |
| RandomForest.ipynb | Random Forest |
| XGBoost.ipynb | XGBoost |

### results/
Contains the model evaluation outputs.

The results include separate dataset performance metrics for:

- AdaBoost
- Lasso Regression
- Random Forest
- XGBoost

## 3. Dataset Preparation

Each dataset is loaded and processed separately.

The preprocessing workflow includes:

1. Loading the dataset.
2. Checking required columns.
3. Handling missing values.
4. Removing duplicate records.
5. Converting target scores into numerical values.
6. Preparing features for machine learning.

The expected core columns in the baseline model include:

- drugname1
- drugname2
- cellline
- score

Column names should be checked and standardized when necessary.

## 4. Feature Engineering

### Baseline Features

The baseline model uses:

- Drug 1 molecular representation.
- Drug 2 molecular representation.
- Cell-line information.

Morgan fingerprints are generated using RDKit to represent the molecular structures of the drugs.

### DRS-Added Features

The enhanced feature approach incorporates additional DRS-derived features alongside the baseline molecular and cell-line features.

The purpose is to investigate whether additional drug-related information improves synergy score prediction.

Target-derived features and data leakage must be excluded during feature selection.

## 5. Machine Learning Models

### XGBoost
Uses gradient boosting to learn relationships between drug-related features and synergy scores.

### Random Forest
Uses an ensemble of decision trees to predict synergy scores.

### AdaBoost
Combines multiple weak learners to improve prediction performance.

### Lasso Regression
Uses L1 regularization to perform regression and feature selection.

## 6. Model Training Workflow

Each dataset is processed separately.

The general training procedure is:

1. Load the dataset.
2. Perform preprocessing.
3. Generate molecular fingerprints.
4. Prepare additional features where applicable.
5. Split the dataset into training and testing sets.
6. Train the selected machine learning algorithm.
7. Generate predictions.
8. Evaluate model performance.
9. Save the evaluation metrics.

The baseline and DRS-enhanced approaches should use consistent data splits for fair comparison.

## 7. Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Purpose | Preferred Direction |
|---|---|---|
| MAE | Measures average absolute prediction error | Lower |
| RMSE | Penalizes larger prediction errors | Lower |
| R² | Measures the proportion of variance explained | Higher |
| Explained Variance | Measures explained variation | Higher |
| Pearson Correlation | Measures linear association | Higher |
| Spearman Correlation | Measures rank association | Higher |

Not all metrics are necessarily calculated by every model.

## 8. Results and Visualization

The `results/` directory contains model evaluation files.

The notebooks also generate graphs for comparing performance across datasets.

These visualizations help identify:

- Differences in prediction errors.
- Differences in model performance.
- Dataset-wise model performance.
- Actual versus predicted synergy scores.

## 9. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/sathvika-panuganti/Synergy-DRS.git
```

### Step 2: Navigate to the project

```bash
cd Synergy-DRS
```

### Step 3: Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost rdkit matplotlib seaborn openpyxl xlrd
```

### Step 4: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Navigate to metadata

Open the required notebook:

- AdaBoost.ipynb
- Lasso.ipynb
- RandomForest.ipynb
- XGBoost.ipynb

### Step 6: Execute the notebook

Run the notebook cells sequentially after checking the dataset paths and required dependencies.

## 10. Important Considerations

- Ensure that dataset paths are correctly configured.
- Verify the target score column before training.
- Avoid data leakage during feature selection.
- Use consistent train-test splits when comparing models.
- Consider group-based validation for testing generalization to unseen drug combinations or cell lines.
- Compare metrics only when target definitions and score scales are compatible.

## 11. Future Improvements

- Additional DRS feature integration.
- Hyperparameter optimization.
- Cross-dataset validation.
- Improved feature selection.
- Advanced predictive modelling.
- Integration with the project frontend.

---

**End of Project Guide**
