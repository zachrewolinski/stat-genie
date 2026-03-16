import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Define dyslexia subset
# Use dyslexia_bin==1 when available; fall back to dyslexia>=1
if "dyslexia_bin" in df.columns:
    dyslex_df = df[df["dyslexia_bin"] == 1].copy()
else:
    dyslex_df = df[df["dyslexia"] >= 1].copy()

# Basic cleaning: drop rows missing key vars
key_vars = ["reader_view", "speed", "uuid"]
for v in key_vars:
    if v not in dyslex_df.columns:
        raise ValueError(f"Missing required column: {v}")

dyslex_df = dyslex_df.dropna(subset=key_vars)

# Ensure reader_view is binary numeric
rv = dyslex_df["reader_view"].astype(int)

dyslex_df = dyslex_df.assign(reader_view=rv)

# Descriptives
n_total = len(dyslex_df)
counts = dyslex_df["reader_view"].value_counts().to_dict()

def group_stats(series):
    return {
        "n": int(series.shape[0]),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=1)) if series.shape[0] > 1 else float("nan"),
    }

speed0 = dyslex_df.loc[dyslex_df["reader_view"] == 0, "speed"].astype(float)
speed1 = dyslex_df.loc[dyslex_df["reader_view"] == 1, "speed"].astype(float)

stats0 = group_stats(speed0)
stats1 = group_stats(speed1)

# Welch t-test on raw speed
welch_t = stats.ttest_ind(speed1, speed0, equal_var=False, nan_policy="omit")

# Mann-Whitney U on raw speed
try:
    mwu = stats.mannwhitneyu(speed1, speed0, alternative="two-sided")
except ValueError:
    mwu = None

# Effect size: Cohen's d
mean_diff = stats1["mean"] - stats0["mean"]
var0 = np.var(speed0, ddof=1) if speed0.shape[0] > 1 else np.nan
var1 = np.var(speed1, ddof=1) if speed1.shape[0] > 1 else np.nan
n0 = speed0.shape[0]
n1 = speed1.shape[0]
pooled_sd = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)) if (n0 + n1 - 2) > 0 else np.nan
cohens_d = mean_diff / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan

# Log-transform speed for skewness
min_speed = dyslex_df["speed"].min()
if min_speed <= 0:
    eps = abs(min_speed) + 1e-6
else:
    eps = 0.0

log_speed = np.log(dyslex_df["speed"] + eps)

dyslex_df = dyslex_df.assign(log_speed=log_speed)

# Welch t-test on log speed
log_speed0 = dyslex_df.loc[dyslex_df["reader_view"] == 0, "log_speed"]
log_speed1 = dyslex_df.loc[dyslex_df["reader_view"] == 1, "log_speed"]
log_welch_t = stats.ttest_ind(log_speed1, log_speed0, equal_var=False, nan_policy="omit")

# Mixed effects model with random intercept for uuid (if possible)
# Control for page_id, num_words, Flesch_Kincaid, device, age, gender, education, language, english_native, retake_trial
# Use log_speed for stability
model_result = None
model_error = None

covariates = [
    "page_id",
    "num_words",
    "Flesch_Kincaid",
    "device",
    "age",
    "gender",
    "education",
    "language",
    "english_native",
    "retake_trial",
]

available_covs = [c for c in covariates if c in dyslex_df.columns]

# Build formula with available covariates
formula_parts = ["reader_view"] + available_covs
formula = "log_speed ~ " + " + ".join([f"C({c})" if dyslex_df[c].dtype == object or dyslex_df[c].dtype.name == "category" else c for c in formula_parts])

# Drop rows with missing model variables
model_cols = ["log_speed", "reader_view", "uuid"] + available_covs
model_df = dyslex_df[model_cols].dropna().copy()

try:
    # MixedLM with random intercept for uuid
    model = smf.mixedlm(formula, model_df, groups=model_df["uuid"])
    model_result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
except Exception as e:
    model_error = str(e)
    # Fallback OLS with cluster-robust SE by uuid
    try:
        ols_model = smf.ols(formula, model_df).fit(cov_type="cluster", cov_kwds={"groups": model_df["uuid"]})
        model_result = ols_model
    except Exception as e2:
        model_error = f"MixedLM failed: {model_error}; OLS failed: {e2}"
        model_result = None

# Extract reader_view effect
rv_effect = None
if model_result is not None:
    if "reader_view" in model_result.params.index:
        rv_effect = {
            "coef": float(model_result.params["reader_view"]),
            "pvalue": float(model_result.pvalues["reader_view"]),
            "ci_low": float(model_result.conf_int().loc["reader_view", 0]) if hasattr(model_result, "conf_int") else float("nan"),
            "ci_high": float(model_result.conf_int().loc["reader_view", 1]) if hasattr(model_result, "conf_int") else float("nan"),
        }

summary = {
    "n_total": int(n_total),
    "counts_by_reader_view": {str(k): int(v) for k, v in counts.items()},
    "speed_stats_reader_view_0": stats0,
    "speed_stats_reader_view_1": stats1,
    "welch_t_raw": {
        "statistic": float(welch_t.statistic),
        "pvalue": float(welch_t.pvalue),
    },
    "mannwhitney_u": None if mwu is None else {"statistic": float(mwu.statistic), "pvalue": float(mwu.pvalue)},
    "cohens_d_raw": float(cohens_d) if cohens_d == cohens_d else None,
    "welch_t_log": {"statistic": float(log_welch_t.statistic), "pvalue": float(log_welch_t.pvalue)},
    "model": {
        "formula": formula,
        "n_model": int(len(model_df)),
        "reader_view_effect": rv_effect,
        "model_type": "MixedLM" if model_result is not None and model_result.__class__.__name__.lower().startswith("mixed") else "OLS_clustered" if model_result is not None else None,
        "model_error": model_error,
    },
}

print(json.dumps(summary, indent=2))

# Determine response scale (0-100) based on evidence
# Heuristic: If reader_view effect is positive and significant across tests, lean Yes.
# If not significant or negative, lean No.

# Default no
response = 25

# Evaluate significance
raw_p = welch_t.pvalue
log_p = log_welch_t.pvalue
model_p = rv_effect["pvalue"] if rv_effect else None

# Determine direction
mean_diff = stats1["mean"] - stats0["mean"]

# Adjust response based on evidence
if (raw_p < 0.05 and log_p < 0.05 and (model_p is not None and model_p < 0.05) and mean_diff > 0):
    response = 80
elif (mean_diff > 0) and ((raw_p < 0.05) or (log_p < 0.05) or (model_p is not None and model_p < 0.05)):
    response = 65
elif (mean_diff < 0) and ((raw_p < 0.05) or (log_p < 0.05) or (model_p is not None and model_p < 0.05)):
    response = 20
else:
    response = 30 if mean_diff > 0 else 15

# Build explanation
explanation = (
    "Analyzed dyslexic participants only (dyslexia_bin==1 when available). "
    f"Total N={n_total} with reader_view counts {counts}. "
    f"Mean speed without reader_view: {stats0['mean']:.2f} (median {stats0['median']:.2f}); "
    f"with reader_view: {stats1['mean']:.2f} (median {stats1['median']:.2f}). "
    f"Welch t-test on raw speed p={raw_p:.4g}; Mann-Whitney p={mwu.pvalue:.4g} if available. "
    f"Log-speed Welch t-test p={log_p:.4g}. "
)

if rv_effect:
    explanation += (
        f"Regression on log(speed) with controls and random/clustered effects (n={len(model_df)}) gave reader_view coef={rv_effect['coef']:.4f} "
        f"(p={rv_effect['pvalue']:.4g}, 95% CI [{rv_effect['ci_low']:.4f}, {rv_effect['ci_high']:.4f}]). "
    )
else:
    explanation += "Regression model for reader_view effect did not converge or was unavailable. "

if response >= 60:
    explanation += "Overall evidence suggests reader_view improves reading speed for dyslexic readers."
else:
    explanation += "Overall evidence does not support a meaningful improvement in reading speed for dyslexic readers when using reader_view."

# Write conclusion.txt
out = {"response": int(response), "explanation": explanation}
with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(out, f)
