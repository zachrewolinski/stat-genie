import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Basic cleaning/validation
# Ensure feature6 is treated as categorical with values yes/no
if df["feature6"].dtype != object:
    df["feature6"] = df["feature6"].astype(str)

# Outcome: frequency of affairs
outcome = df["feature2"]

# Group stats
stats_table = (
    df.groupby("feature6")["feature2"]
    .agg(["count", "mean", "median", "std"])
    .rename_axis("children")
)

# Identify group labels
children_labels = stats_table.index.tolist()

# Expect labels like 'yes' and 'no'
# Ensure consistent ordering: no, yes when possible
if "no" in children_labels and "yes" in children_labels:
    group_no = df[df["feature6"] == "no"]["feature2"].values
    group_yes = df[df["feature6"] == "yes"]["feature2"].values
else:
    # fallback to first/second
    group_no = df[df["feature6"] == children_labels[0]]["feature2"].values
    group_yes = df[df["feature6"] == children_labels[1]]["feature2"].values

# Welch t-test
welch_t = stats.ttest_ind(group_no, group_yes, equal_var=False, nan_policy="omit")

# Mann-Whitney U test
mw = stats.mannwhitneyu(group_no, group_yes, alternative="two-sided")

# Effect size: Cohen's d (using pooled SD)
def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled_sd

# d = mean(no children) - mean(children)
cohen_d = cohens_d(group_no, group_yes)

# Binary any affair
any_affair = (df["feature2"] > 0).astype(int)
df = df.copy()
df["any_affair"] = any_affair

# Chi-square test for association
contingency = pd.crosstab(df["feature6"], df["any_affair"])
chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

# Logistic regression (any affair)
try:
    logit = smf.logit("any_affair ~ C(feature6)", data=df).fit(disp=0)
    logit_p = float(logit.pvalues.filter(like="C(feature6)").iloc[0])
    logit_coef = float(logit.params.filter(like="C(feature6)").iloc[0])
    logit_or = float(np.exp(logit_coef))
except Exception:
    logit = None
    logit_p = np.nan
    logit_coef = np.nan
    logit_or = np.nan

# OLS with controls
# Treat feature3 as categorical gender; others numeric
try:
    ols = smf.ols(
        "feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
        data=df,
    ).fit(cov_type="HC3")
    ols_p = float(ols.pvalues.filter(like="C(feature6)").iloc[0])
    ols_coef = float(ols.params.filter(like="C(feature6)").iloc[0])
except Exception:
    ols = None
    ols_p = np.nan
    ols_coef = np.nan

results = {
    "group_stats": stats_table.to_dict(),
    "mean_no_children": float(np.mean(group_no)),
    "mean_children": float(np.mean(group_yes)),
    "mean_diff_no_minus_yes": float(np.mean(group_no) - np.mean(group_yes)),
    "welch_t_stat": float(welch_t.statistic),
    "welch_t_p": float(welch_t.pvalue),
    "mw_u_stat": float(mw.statistic),
    "mw_u_p": float(mw.pvalue),
    "cohen_d": float(cohen_d),
    "chi2_p": float(chi2_p),
    "logit_p": float(logit_p),
    "logit_or": float(logit_or),
    "ols_p": float(ols_p),
    "ols_coef": float(ols_coef),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
