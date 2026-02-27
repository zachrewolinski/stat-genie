import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv("affairs.csv")

# Identify columns
# feature2: affair frequency
# feature6: children yes/no

# Basic cleaning
_df = _df.copy()

# Normalize children values to lower-case strings
_df["feature6"] = _df["feature6"].astype(str).str.strip().str.lower()

# Keep only valid categories
valid = _df["feature6"].isin(["yes", "no"])
_df = _df[valid].copy()

# Affair frequency numeric
_df["feature2"] = pd.to_numeric(_df["feature2"], errors="coerce")
_df = _df.dropna(subset=["feature2", "feature6"])

# Split groups
with_children = _df[_df["feature6"] == "yes"]["feature2"]
without_children = _df[_df["feature6"] == "no"]["feature2"]

# Summary stats
summary = {
    "n_yes": int(with_children.shape[0]),
    "n_no": int(without_children.shape[0]),
    "mean_yes": float(with_children.mean()),
    "mean_no": float(without_children.mean()),
    "median_yes": float(with_children.median()),
    "median_no": float(without_children.median()),
    "std_yes": float(with_children.std(ddof=1)),
    "std_no": float(without_children.std(ddof=1)),
}

# t-test (Welch)
welch = stats.ttest_ind(with_children, without_children, equal_var=False, nan_policy="omit")
summary["welch_t"] = float(welch.statistic)
summary["welch_p"] = float(welch.pvalue)

# Mann-Whitney U (nonparametric)
try:
    mwu = stats.mannwhitneyu(with_children, without_children, alternative="two-sided")
    summary["mwu_u"] = float(mwu.statistic)
    summary["mwu_p"] = float(mwu.pvalue)
except Exception as exc:
    summary["mwu_error"] = str(exc)

# Effect size: Cohen's d (using pooled SD with Welch correction)
# Use Hedges g small sample correction
n1 = with_children.shape[0]
n2 = without_children.shape[0]
var1 = with_children.var(ddof=1)
var2 = without_children.var(ddof=1)
# pooled SD
sp = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
if sp > 0:
    d = (with_children.mean() - without_children.mean()) / sp
else:
    d = np.nan
# Hedges g correction
if n1 + n2 > 2:
    J = 1 - (3 / (4 * (n1 + n2) - 9))
    g = d * J
else:
    g = np.nan
summary["cohen_d_yes_minus_no"] = float(d)
summary["hedges_g_yes_minus_no"] = float(g)

# Also analyze binary any affair > 0
_df["any_affair"] = (_df["feature2"] > 0).astype(int)

# Proportions
prop_yes = _df[_df["feature6"] == "yes"]["any_affair"].mean()
prop_no = _df[_df["feature6"] == "no"]["any_affair"].mean()
summary["prop_any_yes"] = float(prop_yes)
summary["prop_any_no"] = float(prop_no)

# Two-proportion z-test
count = np.array([
    _df[_df["feature6"] == "yes"]["any_affair"].sum(),
    _df[_df["feature6"] == "no"]["any_affair"].sum(),
])
obs = np.array([
    _df[_df["feature6"] == "yes"].shape[0],
    _df[_df["feature6"] == "no"].shape[0],
])

# statsmodels proportion z-test
z_stat, z_p = sm.stats.proportions_ztest(count, obs, alternative="two-sided")
summary["prop_z"] = float(z_stat)
summary["prop_p"] = float(z_p)
summary["prop_diff_yes_minus_no"] = float(prop_yes - prop_no)

# Logistic regression for any affair ~ children (yes=1)
_df["child_yes"] = (_df["feature6"] == "yes").astype(int)
X = sm.add_constant(_df["child_yes"])
logit = sm.Logit(_df["any_affair"], X).fit(disp=0)
summary["logit_coef_child_yes"] = float(logit.params["child_yes"])
summary["logit_p_child_yes"] = float(logit.pvalues["child_yes"])
summary["logit_odds_ratio"] = float(np.exp(logit.params["child_yes"]))

# Save summary for inspection
with open("analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
