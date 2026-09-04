# %%
DATA_PATH = "pp-2025-part-*.csv"
MODEL_PATH = "artifacts/model.joblib"
METRICS_PATH = "artifacts/metrics.json"
IMPORTANCE_PATH = "artifacts/feature_importance.csv"
RANDOM_STATE = 42
COLUMNS = ["transaction_id", "price", "date", "postcode", "property_type", "old_new", "duration", "paon", "saon", "street", "locality", "town", "district", "county", "ppd_category", "record_status"]
FEATURES = ["property_type", "old_new", "duration", "district", "county", "outcode", "month"]
CAT_FEATURES = ["property_type", "old_new", "duration", "district", "county", "outcode"]

# %%
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# %%
data_files = sorted(Path(".").glob(DATA_PATH))
if not data_files:
    raise FileNotFoundError("No split dataset files found")
df = pd.concat(
    [pd.read_csv(path, header=None, names=COLUMNS, dtype=str) for path in data_files],
    ignore_index=True,
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
print(df.shape)
print(df.head(3))

# %%
print(df.dtypes)
print(df.shape)
print(df.head())

# %%
Path("artifacts").mkdir(exist_ok=True)
plt.figure()
plt.hist(np.log1p(df["price"].dropna()), bins=80)
plt.title("log1p price")
plt.savefig("artifacts/eda_price_hist.png")
plt.close()

# %%
df = df[df["ppd_category"] == "A"].copy()
print(df.shape)

# %%
df = df[df["price"] > 0]
print(df.shape)

# %%
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

# %%
df["outcode"] = df["postcode"].str.split().str[0]
df = df[df["outcode"].notna()]

# %%
df = df.drop_duplicates(subset=["transaction_id"] if df["transaction_id"].is_unique else ["postcode", "date", "price", "paon", "saon", "street"])
print(df.shape)
print(df[FEATURES].isna().sum())

# %%
X = df[FEATURES].copy()

# %%
split_date = df["date"].quantile(0.8)
train_rows, test_rows = df["date"] <= split_date, df["date"] > split_date
cap = df.loc[train_rows, "price"].quantile(0.999)
train_rows &= df["price"] <= cap
X_train, X_test = X.loc[train_rows], X.loc[test_rows]
y_train, y_test = np.log1p(df.loc[train_rows, "price"]), np.log1p(df.loc[test_rows, "price"])
print(X_train.shape, X_test.shape)

# %%
pipe = CatBoostRegressor(loss_function="RMSE", iterations=500, verbose=False, random_seed=RANDOM_STATE)
pipe.fit(X_train, y_train, cat_features=CAT_FEATURES)

# %%
y_pred_log = pipe.predict(X_test)
y_true = np.expm1(y_test)
y_pred = np.expm1(y_pred_log)

# %%
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
baseline = np.expm1(y_train.median())
mdape = np.median(np.abs((y_true - y_pred) / y_true)) * 100
print(mae, rmse, r2, mdape)

# %%
sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(y_true), size=min(5000, len(y_true)), replace=False)
plt.figure()
plt.scatter(y_true.iloc[sample_idx], y_pred[sample_idx], alpha=0.2, s=5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.savefig("artifacts/residuals.png")
plt.close()

# %%
perm_idx = X_test.sample(min(5000, len(X_test)), random_state=RANDOM_STATE).index
perm = permutation_importance(pipe, X_test.loc[perm_idx], y_test.loc[perm_idx], n_repeats=5, random_state=RANDOM_STATE)
importances = perm.importances_mean

# %%
importance_df = pd.DataFrame({"feature": FEATURES, "importance": importances}).sort_values("importance", ascending=False)
print(importance_df)

# %%
top_n = len(FEATURES)
top = importance_df.head(top_n)
plt.figure(figsize=(8, 5))
plt.barh(top["feature"][::-1], top["importance"][::-1])
plt.xlabel("importance")
plt.title("top feature importances")
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png")
plt.close()

# %%
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(pipe, MODEL_PATH)
importance_df.to_csv(IMPORTANCE_PATH, index=False)

# %%
metrics = {"mae": mae, "rmse": rmse, "r2": r2, "mdape_percent": mdape, "baseline_median_mae": mean_absolute_error(y_true, np.full(len(y_true), baseline)), "n_train": len(X_train), "n_test": len(X_test)}
Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2))
print(metrics)

# %%
sample = X_test.iloc[[0]]
pred_gbp = np.expm1(pipe.predict(sample))[0]
actual_gbp = np.expm1(y_test.iloc[0])
print(pred_gbp, actual_gbp)
