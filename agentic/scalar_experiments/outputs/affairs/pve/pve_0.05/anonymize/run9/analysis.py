import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF_PATH = "affairs.csv"
df = pd.read_csv(DF_PATH)

# Variables
outcome = "feature2"
children = "feature6"

# Clean and code
valid = df[[outcome, children, "feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]].copy()
valid = valid.dropna()
valid["children_yes"] = (valid[children].str.lower() == "yes").astype(int)

# Group stats
grp = valid.groupby("children_yes")[outcome].agg(["mean", "std", "count"])

# Two-sample t-test (Welch)
child_yes = valid.loc[valid["children_yes"] == 1, outcome]
child_no = valid.loc[valid["children_yes"] == 0, outcome]

tstat, pval = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy="omit")

# Effect size (Cohen's d with pooled SD)
# Using Hedges g correction for small sample bias
n1, n0 = child_yes.shape[0], child_no.shape[0]
mean1, mean0 = child_yes.mean(), child_no.mean()
var1, var0 = child_yes.var(ddof=1), child_no.var(ddof=1)
sp = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
cohen_d = (mean1 - mean0) / sp if sp > 0 else np.nan
# Hedges g
if n1 + n0 > 2:
    g = cohen_d * (1 - (3 / (4*(n1+n0)-9)))
else:
    g = np.nan

# Nonparametric test (Mann-Whitney U)
try:
    u_stat, u_p = stats.mannwhitneyu(child_yes, child_no, alternative="two-sided")
except Exception:
    u_stat, u_p = np.nan, np.nan

# OLS regression: outcome ~ children + controls
# Controls: gender, age, years married, religiousness, education, occupation, marital happiness
# Use robust SEs (HC3)
model = smf.ols(
    f"{outcome} ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=valid
).fit(cov_type="HC3")

# Extract coefficient for children
coef = model.params.get("children_yes", np.nan)
se = model.bse.get("children_yes", np.nan)
p = model.pvalues.get("children_yes", np.nan)
ci_low, ci_high = model.conf_int().loc["children_yes"].tolist()

# Output summary for manual use
results = {
    "n_total": int(valid.shape[0]),
    "n_children_yes": int(n1),
    "n_children_no": int(n0),
    "group_stats": grp.to_dict(),
    "ttest": {"t": float(tstat), "p": float(pval)},
    "cohen_d": float(cohen_d),
    "hedges_g": float(g),
    "mannwhitney": {"u": float(u_stat), "p": float(u_p)},
    "ols": {
        "coef_children": float(coef),
        "se": float(se),
        "p": float(p),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }
}

print(json.dumps(results, indent=2))
