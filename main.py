# %%
# HM Land Registry 2025 price prediction
DATA_PATH = "pp-2025.csv"
MODEL_PATH = "artifacts/model.joblib"
METRICS_PATH = "artifacts/metrics.json"
IMPORTANCE_PATH = "artifacts/feature_importance.csv"
RANDOM_STATE = 42
COLUMNS = ["transaction_id", "price", "date", "postcode", "property_type", "old_new", "duration", "paon", "saon", "street", "locality", "town", "district", "county", "ppd_category", "record_status"]
FEATURES = ["property_type", "old_new", "duration", "district", "county", "outcode", "month", "quarter"]
OHE_FEATURES = ["property_type", "old_new", "duration", "quarter"]
ORD_FEATURES = ["district", "county", "outcode"]
NUM_FEATURES = ["month"]

# %%
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# %%
df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, dtype=str)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
print(df.shape)
print(df.head(3))

# %%
print(df.dtypes)
print(df.shape)
print(df.head())

# %%
print(df.isna().sum())
print(df["price"].isna().sum())

# %%
print(df["price"].describe())
print(df["price"].quantile([0.25, 0.5, 0.75, 0.99, 0.999]))

# %%
print(df["property_type"].value_counts())
print(df["old_new"].value_counts())
print(df["duration"].value_counts())
print(df["ppd_category"].value_counts())

# %%
Path("artifacts").mkdir(exist_ok=True)
plt.figure()
plt.hist(np.log1p(df["price"].dropna()), bins=80)
plt.title("log1p price")
plt.savefig("artifacts/eda_price_hist.png")
plt.close()

# %%
# category B = linked/additional transactions, not independent sales
df = df[df["ppd_category"] == "A"].copy()
print(df.shape)

# %%
df = df[df["price"] > 0]
print(df.shape)

# %%
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter

# %%
df["outcode"] = df["postcode"].str.split().str[0]
df = df[df["outcode"].notna()]

# %%
cap = df["price"].quantile(0.999)
df = df[df["price"] <= cap]
print(cap)

# %%
df = df.drop_duplicates(subset=["postcode", "date", "price", "paon", "street"])
print(df.shape)
print(df[FEATURES].isna().sum())

# %%
X = df[FEATURES].copy()

# %%
y = np.log1p(df["price"])

# %%
# district/outcode exceed HGBR native categorical limit (255); ordinal encode location instead
preprocessor = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), OHE_FEATURES), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORD_FEATURES), ("num", "passthrough", NUM_FEATURES)])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(X_train.shape, X_test.shape)

# %%
pipe = Pipeline([("prep", preprocessor), ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))])

# %%
pipe.fit(X_train, y_train)

# %%
y_pred_log = pipe.predict(X_test)
y_true = np.expm1(y_test)
y_pred = np.expm1(y_pred_log)

# %%
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
print(mae, rmse, r2)

# %%
sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(y_true), size=min(5000, len(y_true)), replace=False)
plt.figure()
plt.scatter(y_true.iloc[sample_idx], y_pred[sample_idx], alpha=0.2, s=5)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.savefig("artifacts/residuals.png")
plt.close()

# %%
perm_idx = X_test.sample(5000, random_state=RANDOM_STATE).index
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
metrics = {"mae": mae, "rmse": rmse, "r2": r2, "n_train": len(X_train), "n_test": len(X_test)}
Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2))
print(metrics)

# %%
sample = X_test.iloc[[0]]
pred_gbp = np.expm1(pipe.predict(sample))[0]
actual_gbp = np.expm1(y_test.iloc[0])
print(pred_gbp, actual_gbp)
