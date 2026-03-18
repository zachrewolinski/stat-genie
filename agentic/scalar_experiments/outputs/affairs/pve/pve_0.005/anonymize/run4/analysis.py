import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Load data
DATA_PATH = "affairs.csv"
df = pd.read_csv(DATA_PATH)

# Map children indicator
children = df["feature6"].astype(str).str.lower()
df = df.assign(children_yes=(children == "yes").astype(int))

# Outcome
y = df["feature2"].astype(float)

# Group stats
summary = df.groupby("children_yes")["feature2"].agg([
    ("n", "count"),
    ("mean", "mean"),
    ("median", "median"),
    ("std", "std"),
])

# Any-affair indicator summary
any_affair = (y > 0).astype(int)
any_affair_summary = df.assign(any_affair=any_affair).groupby("children_yes")["any_affair"].agg([
    ("n", "count"),
    ("prop_any_affair", "mean"),
])

# Welch t-test for difference in means
vals_yes = df.loc[df.children_yes == 1, "feature2"].astype(float).to_numpy()
vals_no = df.loc[df.children_yes == 0, "feature2"].astype(float).to_numpy()

t_stat, p_val = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy="omit")

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(vals_yes, vals_no, alternative="two-sided")
except ValueError:
    # If all values identical, mannwhitneyu can fail
    u_stat, u_p = np.nan, np.nan

# Cohen's d (Welch)
mean_yes = np.nanmean(vals_yes)
mean_no = np.nanmean(vals_no)
var_yes = np.nanvar(vals_yes, ddof=1)
var_no = np.nanvar(vals_no, ddof=1)
# Pooled standard deviation (unequal sample size)
pooled_sd = np.sqrt(((len(vals_yes) - 1) * var_yes + (len(vals_no) - 1) * var_no) / (len(vals_yes) + len(vals_no) - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# OLS regression: feature2 ~ children_yes
X = sm.add_constant(df["children_yes"])
ols = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")

# Binary outcome: any affair (feature2 > 0)
# Use >0 as a simple indicator for any engagement
logit = None
if any_affair.nunique() > 1:
    logit = sm.Logit(any_affair, X, missing="drop").fit(disp=False)

# Prepare results
results = {
    "summary": summary.reset_index().to_dict(orient="list"),
    "any_affair_summary": any_affair_summary.reset_index().to_dict(orient="list"),
    "welch_t": {"t": t_stat, "p": p_val},
    "mannwhitney": {"u": u_stat, "p": u_p},
    "cohens_d": cohens_d,
    "ols": {
        "coef_children": float(ols.params["children_yes"]),
        "p_children": float(ols.pvalues["children_yes"]),
        "ci_children": [float(x) for x in ols.conf_int().loc["children_yes"].tolist()],
    },
}
if logit is not None:
    results["logit"] = {
        "coef_children": float(logit.params["children_yes"]),
        "p_children": float(logit.pvalues["children_yes"]),
        "ci_children": [float(x) for x in logit.conf_int().loc["children_yes"].tolist()],
    }

with open("analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
