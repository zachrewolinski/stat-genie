import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("affairs.csv")

df = df.copy()

# Ensure feature6 is string lower
df["feature6"] = df["feature6"].astype(str).str.strip().str.lower()

# Outcome
y = df["feature2"].astype(float)

# Grouping
group_yes = df[df["feature6"] == "yes"]["feature2"].astype(float)
group_no = df[df["feature6"] == "no"]["feature2"].astype(float)

# Descriptives
desc = {
    "n_yes": int(group_yes.shape[0]),
    "n_no": int(group_no.shape[0]),
    "mean_yes": float(group_yes.mean()),
    "mean_no": float(group_no.mean()),
    "median_yes": float(group_yes.median()),
    "median_no": float(group_no.median()),
    "sd_yes": float(group_yes.std(ddof=1)),
    "sd_no": float(group_no.std(ddof=1)),
}

# Welch t-test
t_stat, t_p = stats.ttest_ind(group_yes, group_no, equal_var=False, nan_policy="omit")

# Mann-Whitney U (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided")
except ValueError:
    # Fallback if ties or identical values cause issues
    u_stat, u_p = np.nan, np.nan

# Effect size (Cohen's d)
n1, n0 = group_yes.shape[0], group_no.shape[0]
s1, s0 = group_yes.std(ddof=1), group_no.std(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
cohen_d = (group_yes.mean() - group_no.mean()) / pooled_sd if pooled_sd != 0 else np.nan

# Regression with controls
model2 = smf.ols(
    "feature2 ~ C(feature6, Treatment(reference='no')) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit()
coef_yes = model2.params.get("C(feature6, Treatment(reference='no'))[T.yes]", np.nan)
p_yes = model2.pvalues.get("C(feature6, Treatment(reference='no'))[T.yes]", np.nan)

results = {
    "desc": desc,
    "t_test": {"t_stat": float(t_stat), "p_value": float(t_p)},
    "mann_whitney": {"u_stat": float(u_stat), "p_value": float(u_p)},
    "cohen_d": float(cohen_d),
    "regression": {
        "coef_children_yes": float(coef_yes),
        "p_value": float(p_yes),
        "r2": float(model2.rsquared),
    },
}

print(json.dumps(results, indent=2))
