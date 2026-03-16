import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning: drop missing in key columns
key_cols = ["reader_view", "speed", "dyslexia", "dyslexia_bin"]

# create dyslexia indicator: dyslexia_bin if present else dyslexia>=1
if "dyslexia_bin" in df.columns:
    dys = df["dyslexia_bin"]
    dys = dys.astype(float)
else:
    dys = (df["dyslexia"] >= 1).astype(int)

# keep rows with non-missing
mask = df["reader_view"].notna() & df["speed"].notna() & dys.notna()
sub = df.loc[mask].copy()
sub["dyslexia_bin"] = dys.loc[mask].astype(int)

# restrict to dyslexic individuals
sub_dys = sub[sub["dyslexia_bin"] == 1].copy()

# Group stats
rv0 = sub_dys[sub_dys["reader_view"] == 0]["speed"].astype(float)
rv1 = sub_dys[sub_dys["reader_view"] == 1]["speed"].astype(float)

stats_dict = {
    "n_total": len(sub),
    "n_dys": len(sub_dys),
    "n_rv0": rv0.shape[0],
    "n_rv1": rv1.shape[0],
    "mean_rv0": rv0.mean(),
    "mean_rv1": rv1.mean(),
    "median_rv0": rv0.median(),
    "median_rv1": rv1.median(),
    "sd_rv0": rv0.std(ddof=1),
    "sd_rv1": rv1.std(ddof=1),
}

# Welch t-test
if len(rv0) > 1 and len(rv1) > 1:
    t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U
if len(rv0) > 0 and len(rv1) > 0:
    try:
        u_stat, u_p = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    except ValueError:
        u_stat, u_p = np.nan, np.nan
else:
    u_stat, u_p = np.nan, np.nan

# Effect size: Cohen's d for independent samples
if len(rv0) > 1 and len(rv1) > 1:
    mean_diff = rv1.mean() - rv0.mean()
    # pooled SD
    s1 = rv1.std(ddof=1)
    s0 = rv0.std(ddof=1)
    n1 = len(rv1)
    n0 = len(rv0)
    pooled = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2)) if (n1+n0-2) > 0 else np.nan
    cohens_d = mean_diff / pooled if pooled and pooled > 0 else np.nan
else:
    mean_diff = np.nan
    cohens_d = np.nan

# Regression controlling for page and device and language and age, use log(speed) for skewness if positive.
# Add small constant for log in case of zero
sub_dys_reg = sub_dys.copy()
sub_dys_reg = sub_dys_reg[sub_dys_reg["speed"] > 0].copy()
sub_dys_reg["log_speed"] = np.log(sub_dys_reg["speed"])

# Build design matrix
covariates = []
for col in ["page_id", "device", "language", "age"]:
    if col in sub_dys_reg.columns:
        covariates.append(col)

# One-hot encode categoricals
X = sub_dys_reg[["reader_view"] + covariates].copy()
X = pd.get_dummies(X, columns=[c for c in covariates if X[c].dtype == object or str(X[c].dtype).startswith('category')], drop_first=True)

X = sm.add_constant(X, has_constant='add')

model = sm.OLS(sub_dys_reg["log_speed"], X).fit()

# Extract reader_view coefficient
rv_coef = model.params.get("reader_view", np.nan)
rv_p = model.pvalues.get("reader_view", np.nan)

print("STATS", stats_dict)
print("TTEST", t_stat, t_p)
print("MWU", u_stat, u_p)
print("COHENS_D", mean_diff, cohens_d)
print("REG_COEF", rv_coef, rv_p)
