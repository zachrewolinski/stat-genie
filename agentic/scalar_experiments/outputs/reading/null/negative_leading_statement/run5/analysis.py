import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleanup
# Ensure expected columns
required_cols = ["reader_view", "speed", "dyslexia_bin", "num_words", "page_id", "uuid"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

# Filter dyslexic participants
# dyslexia_bin: 1 indicates dyslexia
sub = df[(df["dyslexia_bin"] == 1) & df["speed"].notna() & df["reader_view"].notna()].copy()

# Drop non-positive or extreme? Keep all but filter non-positive speed
sub = sub[sub["speed"] > 0]

# Group stats
stats_by_group = sub.groupby("reader_view")["speed"].agg([
    "count", "mean", "median", "std", "min", "max"
]).reset_index()

# Welch t-test
rv1 = sub[sub["reader_view"] == 1]["speed"].values
rv0 = sub[sub["reader_view"] == 0]["speed"].values

welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Mann-Whitney U (two-sided)
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
except ValueError:
    mwu = None

# Effect size: Cohen's d (Welch)
mean1, mean0 = np.mean(rv1), np.mean(rv0)
var1, var0 = np.var(rv1, ddof=1), np.var(rv0, ddof=1)
# Pooled std for unequal n (still using classic pooled for effect size)
pooled_std = np.sqrt(((len(rv1) - 1) * var1 + (len(rv0) - 1) * var0) / (len(rv1) + len(rv0) - 2))
cohens_d = (mean1 - mean0) / pooled_std if pooled_std > 0 else np.nan

# Robust regression controlling for page and num_words
# speed is often skewed; use log1p
sub["log_speed"] = np.log1p(sub["speed"])
# Use page_id as categorical fixed effect; reader_view as main predictor
# Also include num_words for text length
model = smf.ols("log_speed ~ reader_view + num_words + C(page_id)", data=sub).fit(cov_type="HC3")

# Compute percent change for reader_view on log scale
coef = model.params.get("reader_view", np.nan)
# For log1p, approximate percent change on speed as exp(coef)-1
pct_change = np.expm1(coef) if pd.notna(coef) else np.nan

# Also run a mixed model if possible with participant id. Check if uuid repeats.
# If each uuid is unique per record, mixed model not useful. Use participant inferred if any repeated.
# Try grouping by uuid: if repeated, use random intercept.
use_mixed = sub["uuid"].nunique() < len(sub)
if use_mixed:
    # statsmodels MixedLM may fail if convergence issues; try/except
    try:
        mixed = smf.mixedlm("log_speed ~ reader_view + num_words + C(page_id)", data=sub, groups=sub["uuid"]).fit(reml=False)
        mixed_coef = mixed.params.get("reader_view", np.nan)
        mixed_p = mixed.pvalues.get("reader_view", np.nan)
        mixed_pct = np.expm1(mixed_coef) if pd.notna(mixed_coef) else np.nan
    except Exception:
        mixed = None
        mixed_coef = mixed_p = mixed_pct = np.nan
else:
    mixed = None
    mixed_coef = mixed_p = mixed_pct = np.nan

results = {
    "n_total": int(len(sub)),
    "n_reader_view_on": int(len(rv1)),
    "n_reader_view_off": int(len(rv0)),
    "group_stats": stats_by_group.to_dict(orient="records"),
    "welch_t": {
        "statistic": float(welch_t.statistic),
        "pvalue": float(welch_t.pvalue)
    },
    "mann_whitney": None if mwu is None else {
        "statistic": float(mwu.statistic),
        "pvalue": float(mwu.pvalue)
    },
    "cohens_d": float(cohens_d),
    "ols": {
        "coef_reader_view": float(coef),
        "pvalue_reader_view": float(model.pvalues.get("reader_view", np.nan)),
        "pct_change": float(pct_change)
    },
    "mixedlm": None if mixed is None else {
        "coef_reader_view": float(mixed_coef),
        "pvalue_reader_view": float(mixed_p),
        "pct_change": float(mixed_pct)
    },
    "model_n": int(model.nobs)
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
