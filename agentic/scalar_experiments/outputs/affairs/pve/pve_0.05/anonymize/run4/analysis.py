import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

DATA_PATH = "affairs.csv"

df = pd.read_csv(DATA_PATH)

# Identify columns from metadata
# feature2: affairs frequency, feature6: children yes/no
# Control variables: feature3 gender, feature4 age, feature5 years married, feature7 religiosity,
# feature8 education, feature9 occupation, feature10 marriage rating

# Basic cleaning
# Map children to indicator: yes=1, no=0
children_map = {"yes": 1, "no": 0}
if df["feature6"].dtype == object:
    df["children"] = df["feature6"].str.strip().str.lower().map(children_map)
else:
    # if already encoded numerically, try to map 1/0
    df["children"] = df["feature6"]

# Affairs frequency
affairs = df["feature2"].astype(float)

df["affairs"] = affairs

# Any affair indicator
any_affair = (affairs > 0).astype(int)

df["any_affair"] = any_affair

# Descriptive stats by children
summary = df.groupby("children")["affairs"].agg(["mean", "median", "count"])

# Welch t-test for affairs frequency
children_yes = df.loc[df["children"] == 1, "affairs"]
children_no = df.loc[df["children"] == 0, "affairs"]

t_stat, p_val, dfree = ttest_ind(children_yes, children_no, usevar="unequal")

# Mann-Whitney U as nonparametric check
try:
    from scipy.stats import mannwhitneyu
    u_stat, p_u = mannwhitneyu(children_yes, children_no, alternative="two-sided")
except Exception:
    u_stat, p_u = np.nan, np.nan

# Logistic regression for any affair (unadjusted and adjusted)
# Prepare controls
controls = [
    "feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"
]

# One-hot encode gender
model_df = df[["children", "any_affair", "affairs"] + controls].copy()
model_df = pd.get_dummies(model_df, columns=["feature3"], drop_first=True)

# Unadjusted logit
X_unadj = sm.add_constant(model_df[["children"]])
logit_unadj = sm.Logit(model_df["any_affair"], X_unadj).fit(disp=False)

# Adjusted logit
X_adj = sm.add_constant(model_df.drop(columns=["any_affair", "affairs"]))
logit_adj = sm.Logit(model_df["any_affair"], X_adj).fit(disp=False)

# OLS on raw affairs frequency (adjusted) with robust SE
X_ols = sm.add_constant(model_df.drop(columns=["any_affair", "affairs"]))
ols_adj = sm.OLS(model_df["affairs"], X_ols).fit(cov_type="HC3")

# Cohen's d for difference in means
n1 = len(children_yes)
n0 = len(children_no)
sd1 = children_yes.std(ddof=1)
sd0 = children_no.std(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n0 - 1) * sd0**2) / (n1 + n0 - 2))
cohen_d = (children_yes.mean() - children_no.mean()) / pooled_sd

results = {
    "summary": summary.to_dict(),
    "t_test": {"t_stat": float(t_stat), "p_value": float(p_val), "df": float(dfree)},
    "mannwhitney": {"u_stat": float(u_stat), "p_value": float(p_u)},
    "logit_unadj": {
        "coef_children": float(logit_unadj.params["children"]),
        "p_value_children": float(logit_unadj.pvalues["children"]),
    },
    "logit_adj": {
        "coef_children": float(logit_adj.params["children"]),
        "p_value_children": float(logit_adj.pvalues["children"]),
    },
    "ols_adj": {
        "coef_children": float(ols_adj.params["children"]),
        "p_value_children": float(ols_adj.pvalues["children"]),
    },
    "cohen_d": float(cohen_d),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
