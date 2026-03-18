import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("affairs.csv")

# feature6 is children yes/no
df["children"] = df["feature6"].map({"yes": 1, "no": 0})

# Group stats
groups = df.groupby("children")["feature2"]
mean_no = groups.mean().loc[0]
mean_yes = groups.mean().loc[1]
median_no = groups.median().loc[0]
median_yes = groups.median().loc[1]
n_no = groups.size().loc[0]
n_yes = groups.size().loc[1]

# Welch t-test
t_stat, p_val = stats.ttest_ind(
    df.loc[df["children"] == 0, "feature2"],
    df.loc[df["children"] == 1, "feature2"],
    equal_var=False,
    nan_policy="omit",
)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


d = cohens_d(df.loc[df["children"] == 0, "feature2"], df.loc[df["children"] == 1, "feature2"])

# OLS regression (children indicator)
ols = smf.ols("feature2 ~ children", data=df).fit()
ols_coef = float(ols.params["children"])
ols_p = float(ols.pvalues["children"])

# Logistic regression for any affair (feature2 > 0)
df["any_affair"] = (df["feature2"] > 0).astype(int)
logit = smf.logit("any_affair ~ children", data=df).fit(disp=0)
logit_coef = float(logit.params["children"])
logit_p = float(logit.pvalues["children"])
logit_or = float(np.exp(logit_coef))

results = {
    "n_children_no": int(n_no),
    "n_children_yes": int(n_yes),
    "mean_feature2_no_children": float(mean_no),
    "mean_feature2_children": float(mean_yes),
    "median_feature2_no_children": float(median_no),
    "median_feature2_children": float(median_yes),
    "t_stat": float(t_stat),
    "t_p": float(p_val),
    "cohens_d_no_minus_yes": float(d),
    "ols_coef_children": float(ols_coef),
    "ols_p_children": float(ols_p),
    "logit_coef_children": float(logit_coef),
    "logit_p_children": float(logit_p),
    "logit_or_children": float(logit_or),
}

print(json.dumps(results, indent=2))
