import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "affairs.csv"

df = pd.read_csv(DATA_PATH)

# Map children
# feature6: yes/no
children = df["feature6"].astype(str).str.lower()

df = df.assign(children_yes=(children == "yes").astype(int))

# outcome
y = df["feature2"].astype(float)

# Basic group stats
summary = (
    df.groupby("children_yes")["feature2"]
    .agg(["count", "mean", "median", "std"])
    .rename(index={0: "no_children", 1: "children"})
)

# Two-sample t-test (Welch)
no_children = y[df["children_yes"] == 0]
children_vals = y[df["children_yes"] == 1]

ttest = stats.ttest_ind(no_children, children_vals, equal_var=False, nan_policy="omit")

# Mann-Whitney U test (non-parametric)
mannwhitney = stats.mannwhitneyu(no_children, children_vals, alternative="two-sided")

# Effect size (Cohen's d)
mean_diff = no_children.mean() - children_vals.mean()
pooled_sd = np.sqrt(((no_children.var(ddof=1) + children_vals.var(ddof=1)) / 2.0))
cohen_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan

# Any affair binary outcome
any_affair = (y > 0).astype(int)

df = df.assign(any_affair=any_affair)

# Proportions
prop_any = df.groupby("children_yes")["any_affair"].mean().rename({0: "no_children", 1: "children"})

# Two-proportion z-test
count = np.array([
    df.loc[df["children_yes"] == 0, "any_affair"].sum(),
    df.loc[df["children_yes"] == 1, "any_affair"].sum(),
])
obs = np.array([
    (df["children_yes"] == 0).sum(),
    (df["children_yes"] == 1).sum(),
])

z_stat, z_p = sm.stats.proportions_ztest(count, obs, alternative="two-sided")

# Regression: OLS on feature2 with controls
# Controls: gender (feature3), age (feature4), years married (feature5), religiousness (feature7),
# education (feature8), occupation (feature9), marriage rating (feature10)

# Prepare categorical for gender
# Ensure no spaces in column names

model_ols = smf.ols(
    "feature2 ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit(cov_type="HC3")

# Logistic regression for any affair
model_logit = smf.logit(
    "any_affair ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
    data=df,
).fit(disp=False)

# Odds ratio for children_yes
odds_ratio = np.exp(model_logit.params["children_yes"])
logit_p = model_logit.pvalues["children_yes"]

results = {
    "summary": summary.to_dict(),
    "ttest": {"stat": float(ttest.statistic), "p": float(ttest.pvalue)},
    "mannwhitney": {"stat": float(mannwhitney.statistic), "p": float(mannwhitney.pvalue)},
    "cohen_d": float(cohen_d),
    "prop_any": prop_any.to_dict(),
    "prop_ztest": {"z": float(z_stat), "p": float(z_p)},
    "ols_children": {
        "coef": float(model_ols.params["children_yes"]),
        "p": float(model_ols.pvalues["children_yes"]),
    },
    "logit_children": {
        "coef": float(model_logit.params["children_yes"]),
        "p": float(logit_p),
        "odds_ratio": float(odds_ratio),
    },
}

print(json.dumps(results, indent=2))
