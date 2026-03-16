import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning: drop rows with missing key fields
key_cols = ["speed", "reader_view", "dyslexia_bin", "uuid", "page_id"]
df = df.dropna(subset=key_cols)

# Focus on participants with dyslexia
# dyslexia_bin == 1 indicates dyslexia
sub = df[df["dyslexia_bin"] == 1].copy()

# Remove non-positive speeds if any (for log transform)
sub = sub[sub["speed"] > 0]

# Summary stats by reader_view
summary = (
    sub.groupby("reader_view")
    .agg(
        n=("speed", "size"),
        mean_speed=("speed", "mean"),
        median_speed=("speed", "median"),
        std_speed=("speed", "std"),
    )
)

# Welch t-test on speed
rv1 = sub[sub["reader_view"] == 1]["speed"]
rv0 = sub[sub["reader_view"] == 0]["speed"]

ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Mann-Whitney U test (non-parametric)
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
except Exception:
    mwu = None

# Effect size (Cohen's d) using pooled std (approx)
mean1, mean0 = rv1.mean(), rv0.mean()
std1, std0 = rv1.std(ddof=1), rv0.std(ddof=1)
# pooled sd
pooled_sd = np.sqrt(((len(rv1)-1)*std1**2 + (len(rv0)-1)*std0**2) / (len(rv1)+len(rv0)-2))
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

# Mixed effects model: log(speed) ~ reader_view + page_id, random intercept for uuid
sub["log_speed"] = np.log(sub["speed"])

# Ensure page_id treated as categorical
sub["page_id"] = sub["page_id"].astype("category")

mixed_result = None
try:
    model = smf.mixedlm("log_speed ~ reader_view + C(page_id)", sub, groups=sub["uuid"])
    mixed_result = model.fit(reml=False, method="lbfgs")
except Exception as e:
    mixed_result = e

# Also OLS with cluster-robust SE by uuid for robustness
ols_result = None
try:
    ols_model = smf.ols("log_speed ~ reader_view + C(page_id)", data=sub)
    ols_result = ols_model.fit(cov_type="cluster", cov_kwds={"groups": sub["uuid"]})
except Exception as e:
    ols_result = e

# Print results
print("DYSLEXIA SUBSET SIZE:", len(sub))
print("SUMMARY BY READER_VIEW:\n", summary)
print("\nWelch t-test:", ttest)
if mwu:
    print("Mann-Whitney U:", mwu)
print("Cohen's d (speed, rv1 - rv0):", cohen_d)

if mixed_result is not None:
    print("\nMixedLM result (log_speed ~ reader_view + page_id, random intercept uuid):")
    print(mixed_result.summary())

if ols_result is not None:
    print("\nOLS cluster-robust (log_speed ~ reader_view + page_id):")
    print(ols_result.summary())
