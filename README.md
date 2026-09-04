# House Price Prediction

Predict UK property sale prices from [HM Land Registry Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) (2025, England & Wales).

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```bash
.\.venv\Scripts\python.exe main.py
```

Or open `main.py` and run `# %%` cells in VS Code.

## Output

- `artifacts/model.joblib` — fitted pipeline
- `artifacts/metrics.json` — MAE, RMSE, R2
- `artifacts/feature_importance.csv` — permutation importance
- `artifacts/*.png` — EDA, residuals, importance plots
