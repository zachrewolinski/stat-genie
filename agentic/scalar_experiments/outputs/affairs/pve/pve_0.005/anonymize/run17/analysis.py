import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv("affairs.csv")

# Basic cleaning
# feature6: children yes/no
# feature2: affairs frequency

# Drop rows with missing in relevant columns
relevant = df[["feature2", "feature6", "feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]].dropna()

# Group stats
groups = relevant.groupby("feature6")["feature2"]
summary = groups.agg(["count", "mean", "median", "std", "min", "max"]).to_dict()

# Proportion with any affairs (>0)
relevant["any_affair"] = (relevant["feature2"] > 0).astype(int)
prop_any = relevant.groupby("feature6")["any_affair"].mean().to_dict()

# t-test for mean difference (Welch)
vals_yes = relevant.loc[relevant["feature6"] == "yes", "feature2"]
vals_no = relevant.loc[relevant["feature6"] == "no", "feature2"]

ttest_res = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy="omit")

# Mann-Whitney U (two-sided)
# If all values equal, mannwhitney may fail; guard
try:
    mwu_res = stats.mannwhitneyu(vals_yes, vals_no, alternative="two-sided")
    mwu = {"statistic": float(mwu_res.statistic), "pvalue": float(mwu_res.pvalue)}
except Exception as e:
    mwu = {"error": str(e)}

# OLS with controls
# Encode feature6 as categorical
formula = "feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
ols_model = smf.ols(formula, data=relevant).fit(cov_type="HC3")

# Logistic regression on any_affair
logit_formula = "any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
logit_model = smf.logit(logit_formula, data=relevant).fit(disp=False)

# Extract coefficient for children (yes vs no). The baseline is first category alphabetically? In statsmodels, for C(feature6), it will use "no" as baseline if sorted.
ols_coef = ols_model.params.filter(like="C(feature6)")
ols_p = ols_model.pvalues.filter(like="C(feature6)")

logit_coef = logit_model.params.filter(like="C(feature6)")
logit_p = logit_model.pvalues.filter(like="C(feature6)")

# Odds ratio for logit
logit_or = np.exp(logit_coef)

results = {
    "n": int(relevant.shape[0]),
    "summary": summary,
    "prop_any": prop_any,
    "ttest": {"statistic": float(ttest_res.statistic), "pvalue": float(ttest_res.pvalue)},
    "mannwhitney": mwu,
    "ols_children_coef": ols_coef.to_dict(),
    "ols_children_p": ols_p.to_dict(),
    "logit_children_coef": logit_coef.to_dict(),
    "logit_children_p": logit_p.to_dict(),
    "logit_children_or": logit_or.to_dict(),
}

print(json.dumps(results, indent=2))
