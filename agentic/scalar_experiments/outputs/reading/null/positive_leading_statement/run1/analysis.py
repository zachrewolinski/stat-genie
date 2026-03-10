import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Focus on participants with dyslexia (binary indicator)
if "dyslexia_bin" in df.columns:
    df_dys = df[df["dyslexia_bin"] == 1].copy()
else:
    # fallback: dyslexia > 0
    df_dys = df[df["dyslexia"] > 0].copy()

# Basic counts
summary = {
    "n_rows_total": int(len(df)),
    "n_rows_dyslexia": int(len(df_dys)),
    "n_participants_dyslexia": int(df_dys["uuid"].nunique()),
}

# Remove non-positive or missing speed
speed_col = "speed"
df_dys = df_dys[df_dys[speed_col].notna()].copy()

# Group stats
group_stats = (
    df_dys.groupby("reader_view")[speed_col]
    .agg(["count", "mean", "median", "std"])
    .rename(index={0: "no_reader_view", 1: "reader_view"})
)

summary["group_stats"] = group_stats.to_dict()

# Welch t-test on raw speed
rv = df_dys[df_dys["reader_view"] == 1][speed_col]
no = df_dys[df_dys["reader_view"] == 0][speed_col]

ttest = stats.ttest_ind(rv, no, equal_var=False, nan_policy="omit")
summary["t_test"] = {
    "statistic": float(ttest.statistic),
    "p_value": float(ttest.pvalue),
    "mean_reader_view": float(rv.mean()),
    "mean_no_reader_view": float(no.mean()),
}

# Log-transform to handle skew
# add small constant to avoid log(0)
df_dys["log_speed"] = np.log1p(df_dys[speed_col])

# OLS with cluster-robust SE by participant
# Include controls likely related to reading speed
controls = [
    "C(page_id)",
    "num_words",
    "C(device)",
    "age",
    "correct_rate",
    "retake_trial",
    "Flesch_Kincaid",
]
if "english_native" in df_dys.columns:
    controls.append("C(english_native)")

formula = "log_speed ~ reader_view + " + " + ".join(controls)

# Drop rows with missing in model vars
model_df = df_dys.copy()
model_df = model_df.dropna(subset=[
    "log_speed",
    "reader_view",
    "page_id",
    "num_words",
    "device",
    "age",
    "correct_rate",
    "retake_trial",
    "Flesch_Kincaid",
])
if "english_native" in model_df.columns:
    model_df = model_df.dropna(subset=["english_native"])

model = smf.ols(formula=formula, data=model_df).fit(
    cov_type="cluster", cov_kwds={"groups": model_df["uuid"]}
)

coef = model.params.get("reader_view", np.nan)
se = model.bse.get("reader_view", np.nan)
pval = model.pvalues.get("reader_view", np.nan)

summary["regression_log_speed"] = {
    "coef_reader_view": float(coef),
    "se_reader_view": float(se),
    "p_value_reader_view": float(pval),
    "n_obs": int(model.nobs),
    "n_clusters": int(model_df["uuid"].nunique()),
}

# Percent change approximation from log coefficient
if np.isfinite(coef):
    summary["regression_log_speed"]["pct_change"] = float((np.exp(coef) - 1) * 100.0)

# Save summary for inspection
with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
