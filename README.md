# Predictive Modelling & Classification with XGBoost

A beginner-friendly, end-to-end machine-learning project. It trains and evaluates
one **classification** model and one **regression** model, compares both with
simple baselines, and reports metrics using data the models did not train on.

## What you will learn

- How classification differs from regression
- How to split data into training and test sets
- How a scikit-learn pipeline applies missing-value handling, scaling, and a model
- How to compare XGBoost with a simple baseline
- How to read precision, recall, F1, RMSE, and R²

## Datasets

Both datasets are included in scikit-learn, so the project does not download
data or require API keys:

| Task | Dataset | Rows | Features | What the model predicts |
| --- | --- | ---: | ---: | --- |
| Classification | Breast Cancer Wisconsin Diagnostic | 569 | 30 | Whether a tumour is malignant |
| Regression | Diabetes | 442 | 10 | A quantitative disease-progression measure |

These are intentionally small teaching datasets, not a 10,000-row dataset.
The classification dataset is provided by scikit-learn from the UCI Machine
Learning Repository. The regression dataset is also distributed with
scikit-learn. Read the dataset descriptions in the notebook before interpreting
the results; this is an educational exercise, not a medical tool.

## Project structure

```text
predictive-modelling-xgboost/
├── notebooks/
│   └── xgboost_walkthrough.ipynb   # Guided, cell-by-cell explanation
├── outputs/                        # Created when you run the project
│   └── test_metrics.csv
├── src/
│   ├── modeling.py                 # Data, pipelines, and evaluation
│   └── run_project.py              # Command-line entry point
├── requirements.txt
└── README.md
```

## Setup on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use the environment's
Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the project

Run both experiments from PowerShell:

```powershell
python -m src.run_project
```

Or open `notebooks/xgboost_walkthrough.ipynb` in VS Code or Jupyter, select the
`.venv` Python interpreter as the notebook kernel, and run the cells from top
to bottom. The notebook introduces each step and reuses the same functions as
the command-line run.

The script prints classification and regression metrics and writes the same
results to `outputs/test_metrics.csv`.

### Example results

One run in the included environment, using the fixed 20% test splits and
`random_state=42`, produced:

| Task | Model | Precision | Recall | F1 | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Classification | Dummy baseline | 0.000 | 0.000 | 0.000 | — |
| Classification | XGBoost | 1.000 | 0.929 | 0.963 | — |
| Regression | Dummy baseline | — | — | — | 73.222 |
| Regression | XGBoost | — | — | — | 52.489 |

The regression RMSE was 28.3% lower than the dummy baseline on this split.
Classification F1 increased by 0.963 points (96.3 percentage points); a
relative percentage improvement is undefined because the baseline F1 is zero.
Your results may vary slightly with different library versions.

## How to read the metrics

### Classification

The positive class is **malignant**. Precision answers “Of the tumours predicted
malignant, how many were malignant?” Recall answers “Of the malignant tumours,
how many did the model find?” F1 balances precision and recall. Accuracy is
included, but can be misleading when classes are imbalanced.

### Regression

RMSE (root mean squared error) measures prediction error in the dataset's target
units; **lower is better**. R² is a relative measure of fit; a value closer to
1 is generally better, while a negative value means the model did worse than
predicting the test-set mean.

The dummy baselines make simple predictions without learning patterns: the
most common class for classification and the training-set mean for regression.
Compare the XGBoost rows with those baselines rather than treating a single
score as proof that a model is good.

## Notes about preprocessing and reproducibility

Each model uses a scikit-learn pipeline with median imputation and standard
scaling. The included datasets have no missing feature values; imputation is
shown as a reusable pipeline step for beginner practice. Tree-based XGBoost
does not generally require scaling, but it is included here to demonstrate
how preprocessing can be kept together with a model. The split and model seeds
are fixed so reruns are reproducible with the same library versions.

This project is educational and does not provide medical advice or a validated
clinical prediction.
