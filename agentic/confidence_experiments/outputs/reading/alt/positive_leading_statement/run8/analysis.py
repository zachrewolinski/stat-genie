import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Prefer dyslexia_bin if available; fallback to dyslexia > 0
if "dyslexia_bin" in df.columns:
    dyslexic = df[df["dyslexia_bin"] == 1].copy()
else:
    dyslexic = df[df["dyslexia"] > 0].copy()

# Clean speed: drop missing or non-positive speeds
speed_col = "speed"
if speed_col not in dyslexic.columns:
    raise ValueError("speed column not found")

# Remove non-positive or missing speeds
before_n = len(dyslexic)
dyslexic = dyslexic.dropna(subset=[speed_col, "reader_view"])
dyslexic = dyslexic[dyslexic[speed_col] > 0]

after_n = len(dyslexic)

# Basic counts
n_obs = len(dyslexic)
unique_participants = dyslexic["uuid"].nunique() if "uuid" in dyslexic.columns else None

# Group summaries
summary = dyslexic.groupby("reader_view")[speed_col].agg(
    ["count", "mean", "median", "std"]
)

# Difference in means (reader_view=1 - reader_view=0)
if set(dyslexic["reader_view"].unique()) >= {0, 1}:
    mean_rv1 = summary.loc[1, "mean"]
    mean_rv0 = summary.loc[0, "mean"]
    diff_mean = mean_rv1 - mean_rv0
else:
    mean_rv1 = mean_rv0 = diff_mean = np.nan

# Effect size (Hedges g) for independent groups
# Note: may overstate due to repeated measures, but provides scale.
if set(dyslexic["reader_view"].unique()) >= {0, 1}:
    g0 = dyslexic[dyslexic["reader_view"] == 0][speed_col]
    g1 = dyslexic[dyslexic["reader_view"] == 1][speed_col]
    n0, n1 = len(g0), len(g1)
    s0, s1 = g0.std(ddof=1), g1.std(ddof=1)
    pooled_sd = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    cohen_d = (g1.mean() - g0.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    # Hedges g correction
    J = 1 - (3 / (4 * (n0 + n1) - 9)) if (n0 + n1) > 2 else 1
    hedges_g = cohen_d * J
else:
    hedges_g = np.nan

# Welch t-test (between groups) on raw speed
if set(dyslexic["reader_view"].unique()) >= {0, 1}:
    t_stat, p_val = stats.ttest_ind(
        dyslexic[dyslexic["reader_view"] == 1][speed_col],
        dyslexic[dyslexic["reader_view"] == 0][speed_col],
        equal_var=False,
    )
else:
    t_stat = p_val = np.nan

# Mixed effects model to account for repeated measures by participant
# Use log speed to reduce skew.
model_result = None
mixed_p = np.nan
mixed_coef = np.nan

if "uuid" in dyslexic.columns and set(dyslexic["reader_view"].unique()) >= {0, 1}:
    dyslexic["log_speed"] = np.log(dyslexic[speed_col])
    # Fixed effects: reader_view plus controls for page_id and num_words (if available)
    # Page_id may capture content differences; num_words accounts for length
    fixed_terms = ["reader_view"]
    if "num_words" in dyslexic.columns:
        fixed_terms.append("num_words")
    if "page_id" in dyslexic.columns:
        fixed_terms.append("C(page_id)")

    formula = "log_speed ~ " + " + ".join(fixed_terms)

    try:
        md = smf.mixedlm(formula, dyslexic, groups=dyslexic["uuid"])
        mdf = md.fit(reml=False, method="lbfgs")
        model_result = mdf
        mixed_coef = float(mdf.params.get("reader_view", np.nan))
        mixed_p = float(mdf.pvalues.get("reader_view", np.nan))
    except Exception:
        # Fall back to OLS with clustered SEs by participant
        try:
            ols = smf.ols(formula, data=dyslexic).fit(
                cov_type="cluster", cov_kwds={"groups": dyslexic["uuid"]}
            )
            mixed_coef = float(ols.params.get("reader_view", np.nan))
            mixed_p = float(ols.pvalues.get("reader_view", np.nan))
        except Exception:
            pass

# Output key stats for manual reasoning
output = {
    "n_obs": n_obs,
    "unique_participants": unique_participants,
    "mean_rv1": float(mean_rv1) if np.isfinite(mean_rv1) else None,
    "mean_rv0": float(mean_rv0) if np.isfinite(mean_rv0) else None,
    "diff_mean": float(diff_mean) if np.isfinite(diff_mean) else None,
    "hedges_g": float(hedges_g) if np.isfinite(hedges_g) else None,
    "welch_t": float(t_stat) if np.isfinite(t_stat) else None,
    "welch_p": float(p_val) if np.isfinite(p_val) else None,
    "mixed_coef_log": mixed_coef if np.isfinite(mixed_coef) else None,
    "mixed_p": mixed_p if np.isfinite(mixed_p) else None,
    "summary": summary.reset_index().to_dict(orient="records"),
    "dropped_rows": int(before_n - after_n),
}

print(json.dumps(output, indent=2))
