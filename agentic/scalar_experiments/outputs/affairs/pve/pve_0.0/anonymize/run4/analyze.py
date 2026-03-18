import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Columns
# feature2: affairs frequency (numeric)
# feature6: children yes/no

# Clean/encode
# Ensure feature6 lower case
children = df["feature6"].astype(str).str.lower().str.strip()

df = df.copy()
df["children_yes"] = (children == "yes").astype(int)

# Outcome
y = pd.to_numeric(df["feature2"], errors="coerce")

# Drop missing
mask = y.notna() & df["children_yes"].notna()
df = df.loc[mask].copy()
y = df["feature2"].astype(float)

# Summary stats
summary = df.groupby("children_yes")["feature2"].agg([
    "count", "mean", "median", "std"
]).rename(index={0: "no_children", 1: "children"})

# Proportion any affairs (>0)
prop_any = df.groupby("children_yes")["feature2"].apply(lambda s: (s > 0).mean())

# t-test (Welch)
no = df.loc[df["children_yes"] == 0, "feature2"].astype(float)
yes = df.loc[df["children_yes"] == 1, "feature2"].astype(float)

t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False, nan_policy="omit")

# Mann-Whitney U
try:
    mw_stat, mw_p = stats.mannwhitneyu(yes, no, alternative="two-sided")
except Exception:
    mw_stat, mw_p = np.nan, np.nan

# OLS regression with robust SE
X = sm.add_constant(df["children_yes"].astype(float))
ols_model = sm.OLS(y, X).fit(cov_type="HC1")

# Logistic regression for any affairs
any_affair = (y > 0).astype(int)
logit_model = sm.Logit(any_affair, X).fit(disp=False)

# Extract key stats
results = {
    "summary": summary.to_dict(),
    "prop_any": prop_any.to_dict(),
    "t_test": {"t": float(t_stat), "p": float(t_p)},
    "mannwhitney": {"u": float(mw_stat), "p": float(mw_p)},
    "ols": {
        "coef_children": float(ols_model.params["children_yes"]),
        "p_children": float(ols_model.pvalues["children_yes"]),
        "coef_const": float(ols_model.params["const"])
    },
    "logit": {
        "coef_children": float(logit_model.params["children_yes"]),
        "p_children": float(logit_model.pvalues["children_yes"])
    }
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

