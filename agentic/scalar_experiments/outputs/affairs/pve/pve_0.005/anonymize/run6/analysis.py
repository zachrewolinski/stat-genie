import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "affairs.csv"

df = pd.read_csv(DATA_PATH)

# Map children to binary
child_map = {"yes": 1, "no": 0}
df["children"] = df["feature6"].map(child_map)

# Basic group stats
summary = df.groupby("children")["feature2"].agg(["count", "mean", "std", "median"]).rename(index={0: "no", 1: "yes"})

# Welch's t-test
no_vals = df.loc[df["children"] == 0, "feature2"].to_numpy()
yes_vals = df.loc[df["children"] == 1, "feature2"].to_numpy()

t_stat, p_val = stats.ttest_ind(yes_vals, no_vals, equal_var=False, nan_policy="omit")

# Mann-Whitney U (non-parametric)
try:
    u_stat, u_p = stats.mannwhitneyu(yes_vals, no_vals, alternative="two-sided")
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Cohen's d (yes - no)
mean_yes = np.mean(yes_vals)
mean_no = np.mean(no_vals)
var_yes = np.var(yes_vals, ddof=1)
var_no = np.var(no_vals, ddof=1)

d = (mean_yes - mean_no) / np.sqrt(((len(yes_vals) - 1) * var_yes + (len(no_vals) - 1) * var_no) / (len(yes_vals) + len(no_vals) - 2))

# OLS without controls
model_simple = smf.ols("feature2 ~ children", data=df).fit(cov_type="HC3")

# OLS with controls
model_controls = smf.ols(
    "feature2 ~ children + feature3 + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit(cov_type="HC3")

results = {
    "summary": summary.to_dict(),
    "t_test": {"t_stat": float(t_stat), "p_value": float(p_val)},
    "mannwhitney": {"u_stat": float(u_stat), "p_value": float(u_p)},
    "cohens_d": float(d),
    "ols_simple": {
        "coef_children": float(model_simple.params.get("children", np.nan)),
        "p_value_children": float(model_simple.pvalues.get("children", np.nan)),
    },
    "ols_controls": {
        "coef_children": float(model_controls.params.get("children", np.nan)),
        "p_value_children": float(model_controls.pvalues.get("children", np.nan)),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
