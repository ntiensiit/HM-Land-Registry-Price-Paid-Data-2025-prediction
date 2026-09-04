# House Price Prediction

Predict UK property sale prices from [HM Land Registry Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) (2025, England & Wales).

## Dataset preview

The dataset is split into `pp-2025-part-01.csv` and `pp-2025-part-02.csv`. The files have no header row and use the following columns:

| Column | Description | Example |
|---|---|---|
| `transaction_id` | Unique Land Registry transaction identifier | `{42C129E4-C259-60A9-E063-4804A8C0C25D}` |
| `price` | Sale price in GBP | `635000` |
| `date` | Transaction date | `2025-05-14` |
| `postcode` | Property postcode | `LN11 8GN` |
| `property_type` | D: detached, S: semi-detached, T: terraced, F: flat | `D` |
| `old_new` | Y: new build, N: established property | `N` |
| `duration` | F: freehold, L: leasehold | `F` |
| `paon` / `saon` | Primary and secondary address details | `2` / empty |
| `street` | Street name | `KENWICK VIEW` |
| `locality` | Locality name | empty |
| `town` | Town or city | `LOUTH` |
| `district` | Local authority district | `EAST LINDSEY` |
| `county` | County | `LINCOLNSHIRE` |
| `ppd_category` | A: individual sale, B: additional or linked transaction | `A` |
| `record_status` | Record status code | `A` |

Example row:

```text
"{42C129E4-C259-60A9-E063-4804A8C0C25D}","635000","2025-05-14 00:00","LN11 8GN","D","N","F","2","","KENWICK VIEW","","LOUTH","EAST LINDSEY","LINCOLNSHIRE","A","A"
```

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
- `artifacts/metrics.json` — MAE, RMSE, R2, MdAPE, and median-baseline MAE
- `artifacts/feature_importance.csv` — permutation importance
- `artifacts/*.png` — EDA, residuals, importance plots
