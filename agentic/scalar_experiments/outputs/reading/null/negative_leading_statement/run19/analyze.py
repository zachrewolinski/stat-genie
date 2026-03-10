import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import pingouin as pg

# Load data
csv_path = "reading.csv"
df = pd.read_csv(csv_path)

# Focus on dyslexia participants
# Use dyslexia_bin if available; otherwise dyslexia>0
if "dyslexia_bin" in df.columns:
    dys_df = df[df["dyslexia_bin"] == 1].copy()
else:
    dys_df = df[df["dyslexia"] > 0].copy()

# Ensure reader_view is binary
# Drop rows with missing critical values
cols_needed = ["reader_view", "speed"]
dys_df = dys_df.dropna(subset=cols_needed)

# Basic group stats
rv1 = dys_df[dys_df["reader_view"] == 1]["speed"].astype(float)
rv0 = dys_df[dys_df["reader_view"] == 0]["speed"].astype(float)

summary = {
    "n_total": int(dys_df.shape[0]),
    "n_rv1": int(rv1.shape[0]),
    "n_rv0": int(rv0.shape[0]),
    "mean_rv1": float(rv1.mean()) if rv1.shape[0] else np.nan,
    "mean_rv0": float(rv0.mean()) if rv0.shape[0] else np.nan,
    "median_rv1": float(rv1.median()) if rv1.shape[0] else np.nan,
    "median_rv0": float(rv0.median()) if rv0.shape[0] else np.nan,
    "std_rv1": float(rv1.std(ddof=1)) if rv1.shape[0] else np.nan,
    "std_rv0": float(rv0.std(ddof=1)) if rv0.shape[0] else np.nan,
}

# Welch t-test on raw speed
if rv1.shape[0] > 1 and rv0.shape[0] > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
else:
    t_stat, p_val = np.nan, np.nan

# Nonparametric Mann-Whitney U (two-sided)
if rv1.shape[0] > 0 and rv0.shape[0] > 0:
    try:
        u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
    except ValueError:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

# Effect size (Cohen's d, Hedges g)
if rv1.shape[0] > 1 and rv0.shape[0] > 1:
    d = pg.compute_effsize(rv1, rv0, eftype="cohen")
    g = pg.compute_effsize(rv1, rv0, eftype="hedges")
else:
    d, g = np.nan, np.nan

# Log transform for skewness (add small constant to avoid log(0))
# Speeds appear positive; still add epsilon for safety
epsilon = 1e-6
log_speed = np.log(dys_df["speed"].astype(float) + epsilon)

# Regression: log(speed) ~ reader_view + controls
# Use available controls to reduce confounding
# Build formula with categorical covariates if present
candidate_cats = ["page_id", "device", "education", "gender", "language", "english_native", "retake_trial"]
formula_terms = ["reader_view"]

for col in candidate_cats:
    if col in dys_df.columns:
        formula_terms.append(f"C({col})")

candidate_nums = ["age", "num_words", "Flesch_Kincaid", "img_width", "correct_rate", "scrolling_time"]
for col in candidate_nums:
    if col in dys_df.columns:
        formula_terms.append(col)

formula = "log_speed ~ " + " + ".join(formula_terms)

reg_results = None
try:
    dys_df = dys_df.copy()
    dys_df["log_speed"] = log_speed
    model = smf.ols(formula=formula, data=dys_df).fit(cov_type="HC3")
    reg_results = {
        "coef_reader_view": float(model.params.get("reader_view", np.nan)),
        "se_reader_view": float(model.bse.get("reader_view", np.nan)),
        "p_reader_view": float(model.pvalues.get("reader_view", np.nan)),
        "n_obs": int(model.nobs),
        "r2": float(model.rsquared),
        "formula": formula,
    }
except Exception as e:
    reg_results = {"error": str(e), "formula": formula}

output = {
    "summary": summary,
    "welch_t": {"t": float(t_stat), "p": float(p_val)},
    "mannwhitney": {"u": float(u_stat), "p": float(p_u)},
    "effect_sizes": {"cohen_d": float(d), "hedges_g": float(g)},
    "regression": reg_results,
}

print(json.dumps(output, indent=2))
