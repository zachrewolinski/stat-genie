import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv("affairs.csv")

# Focus on affair frequency and children indicator
cols = ["feature2", "feature6", "feature3", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]
df = df[cols].dropna()

df["children_yes"] = (df["feature6"].str.lower() == "yes").astype(int)

# Group stats
group_yes = df[df["children_yes"] == 1]["feature2"]
group_no = df[df["children_yes"] == 0]["feature2"]

stats_summary = {
    "n_yes": int(group_yes.shape[0]),
    "n_no": int(group_no.shape[0]),
    "mean_yes": float(group_yes.mean()),
    "mean_no": float(group_no.mean()),
    "median_yes": float(group_yes.median()),
    "median_no": float(group_no.median()),
    "std_yes": float(group_yes.std(ddof=1)),
    "std_no": float(group_no.std(ddof=1)),
}

# Welch t-test for means
welch = stats.ttest_ind(group_yes, group_no, equal_var=False, nan_policy="omit")

# Mann-Whitney U test (two-sided)
# Use method='asymptotic' for ties/large sample
mw = stats.mannwhitneyu(group_yes, group_no, alternative="two-sided", method="asymptotic")

# Cohen's d (pooled SD)
n1, n2 = stats_summary["n_yes"], stats_summary["n_no"]
var1, var2 = group_yes.var(ddof=1), group_no.var(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
cohens_d = (stats_summary["mean_yes"] - stats_summary["mean_no"]) / pooled_sd if pooled_sd > 0 else np.nan

# Any-affair proportion comparison
any_yes = (group_yes > 0).sum()
any_no = (group_no > 0).sum()
prop_yes = any_yes / n1
prop_no = any_no / n2

zstat, p_prop = proportions_ztest([any_yes, any_no], [n1, n2])

# OLS with controls (robust SE)
df["male"] = (df["feature3"].str.lower() == "male").astype(int)
ols = smf.ols(
    "feature2 ~ children_yes + male + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit(cov_type="HC3")

# Logistic regression for any affair (may fail if separation)
df["affair_any"] = (df["feature2"] > 0).astype(int)
logit_result = None
try:
    logit = smf.logit(
        "affair_any ~ children_yes + male + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
        data=df,
    ).fit(disp=0)
    logit_result = {
        "coef": float(logit.params["children_yes"]),
        "pvalue": float(logit.pvalues["children_yes"]),
        "odds_ratio": float(np.exp(logit.params["children_yes"])),
    }
except Exception as e:
    logit_result = {"error": str(e)}

results = {
    "summary": stats_summary,
    "welch_t": {"stat": float(welch.statistic), "pvalue": float(welch.pvalue)},
    "mannwhitney": {"stat": float(mw.statistic), "pvalue": float(mw.pvalue)},
    "cohens_d": float(cohens_d),
    "proportions": {
        "any_yes": int(any_yes),
        "any_no": int(any_no),
        "prop_yes": float(prop_yes),
        "prop_no": float(prop_no),
        "zstat": float(zstat),
        "pvalue": float(p_prop),
        "risk_diff": float(prop_yes - prop_no),
    },
    "ols_children": {
        "coef": float(ols.params["children_yes"]),
        "pvalue": float(ols.pvalues["children_yes"]),
    },
    "logit_children": logit_result,
}

print(json.dumps(results, indent=2))
