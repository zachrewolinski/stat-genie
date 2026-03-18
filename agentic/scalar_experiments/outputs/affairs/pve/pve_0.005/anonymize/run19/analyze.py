import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
df = pd.read_csv("affairs.csv")

# feature2: affairs frequency (noisy numeric)
# feature6: children yes/no
children = df["feature6"].astype(str)
affairs = df["feature2"].astype(float)

# Basic group stats
stats_by = df.groupby("feature6")["feature2"].agg(["count", "mean", "median", "std"])

# Proportion with any affair (>0) on noisy scale
df["any_affair"] = (affairs > 0).astype(int)
prop_any = df.groupby("feature6")["any_affair"].mean()
counts_any = df.groupby("feature6")["any_affair"].sum()

# Two-sample tests
vals_yes = affairs[children == "yes"]
vals_no = affairs[children == "no"]

mw = stats.mannwhitneyu(vals_yes, vals_no, alternative="two-sided")
ttest = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy="omit")

# Difference in proportion any affair
from statsmodels.stats.proportion import proportions_ztest

count = np.array([counts_any.get("yes", 0), counts_any.get("no", 0)])
obs = np.array([len(vals_yes), len(vals_no)])
prop_test = proportions_ztest(count, obs)

# Linear regression: affairs ~ children (yes vs no)
df["children_yes"] = (children == "yes").astype(int)
ols_model = sm.OLS(affairs, sm.add_constant(df["children_yes"]))
ols_res = ols_model.fit()

# Logistic regression: any_affair ~ children (yes vs no)
logit_model = sm.Logit(df["any_affair"], sm.add_constant(df["children_yes"]))
logit_res = logit_model.fit(disp=False)

# Collect results
out = {
    "stats_by": stats_by.to_dict(),
    "prop_any": prop_any.to_dict(),
    "mw": {"statistic": mw.statistic, "pvalue": mw.pvalue},
    "ttest": {"statistic": ttest.statistic, "pvalue": ttest.pvalue},
    "prop_test": {"statistic": float(prop_test[0]), "pvalue": float(prop_test[1])},
    "ols_coef": float(ols_res.params["children_yes"]),
    "ols_pvalue": float(ols_res.pvalues["children_yes"]),
    "logit_coef": float(logit_res.params["children_yes"]),
    "logit_pvalue": float(logit_res.pvalues["children_yes"]),
    "logit_or": float(np.exp(logit_res.params["children_yes"])),
    "n_yes": int(len(vals_yes)),
    "n_no": int(len(vals_no)),
}

print(json.dumps(out, indent=2))
