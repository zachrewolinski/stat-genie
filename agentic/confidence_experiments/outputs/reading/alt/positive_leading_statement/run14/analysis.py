import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Identify dyslexia group
if "dyslexia_bin" in df.columns:
    dys = df[df["dyslexia_bin"] == 1].copy()
else:
    dys = df[df["dyslexia"].isin([1, 2])].copy()

# Basic cleaning
cols_needed = ["reader_view", "speed", "uuid"]
for c in cols_needed:
    if c not in dys.columns:
        raise ValueError(f"Missing column {c}")

dys = dys.dropna(subset=["reader_view", "speed"])

# Ensure reader_view binary
rv = dys["reader_view"].astype(int)

dys = dys.assign(reader_view=rv)

# Group stats
summary = dys.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).rename(index={0: "no_reader_view", 1: "reader_view"})

# T-test on raw speed (Welch)
rv1 = dys[dys["reader_view"] == 1]["speed"]
rv0 = dys[dys["reader_view"] == 0]["speed"]

ttest_raw = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Log transform for skew
# Add small constant in case of zeros (min should be >0, but safe)
log_speed = np.log(dys["speed"].clip(lower=1e-6))

dys = dys.assign(log_speed=log_speed)

rv1_log = dys[dys["reader_view"] == 1]["log_speed"]
rv0_log = dys[dys["reader_view"] == 0]["log_speed"]

ttest_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy="omit")

# Regression with covariates (clustered SE by uuid)
# Build formula with available covariates
covariates = []
for col in ["page_id", "num_words", "device", "age", "gender", "education", "language", "english_native", "Flesch_Kincaid", "retake_trial"]:
    if col in dys.columns:
        if dys[col].dtype == "object" or str(dys[col].dtype).startswith("category"):
            covariates.append(f"C({col})")
        else:
            covariates.append(col)

formula = "log_speed ~ reader_view"
if covariates:
    formula += " + " + " + ".join(covariates)

# Drop rows with missing covariates used in formula to align cluster groups
model_data = dys.copy()
needed_cols = ["log_speed", "reader_view", "uuid"]
for col in ["page_id", "num_words", "device", "age", "gender", "education", "language", "english_native", "Flesch_Kincaid", "retake_trial"]:
    if col in model_data.columns:
        needed_cols.append(col)
model_data = model_data.dropna(subset=needed_cols)

model = smf.ols(formula, data=model_data).fit(
    cov_type="cluster",
    cov_kwds={"groups": model_data["uuid"]},
)

coef = model.params.get("reader_view", np.nan)
se = model.bse.get("reader_view", np.nan)
# two-sided p-value
pval = model.pvalues.get("reader_view", np.nan)

# Convert log coefficient to percent change
pct_change = (np.exp(coef) - 1) * 100 if pd.notnull(coef) else np.nan

results = {
    "n_dyslexia": int(dys.shape[0]),
    "n_unique_participants": int(dys["uuid"].nunique()),
    "summary_speed": summary.reset_index().to_dict(orient="records"),
    "ttest_raw": {
        "statistic": float(ttest_raw.statistic),
        "pvalue": float(ttest_raw.pvalue)
    },
    "ttest_log": {
        "statistic": float(ttest_log.statistic),
        "pvalue": float(ttest_log.pvalue)
    },
    "regression": {
        "coef_log_speed_reader_view": float(coef),
        "se": float(se),
        "pvalue": float(pval),
        "pct_change": float(pct_change)
    }
}

print(json.dumps(results, indent=2))
