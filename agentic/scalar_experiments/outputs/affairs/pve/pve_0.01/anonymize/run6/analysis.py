import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "affairs.csv"
df = pd.read_csv(csv_path)

# Identify key columns
# From info.json: feature2 = affairs frequency, feature6 = children yes/no
# feature3 = gender, feature4 age, feature5 years married, feature7 relig, feature8 education, feature9 occupation, feature10 marriage rating

# Clean/prepare
# Ensure feature6 is binary indicator: 1 if children yes, 0 if no
if df["feature6"].dtype == object:
    df["children_yes"] = (df["feature6"].str.lower() == "yes").astype(int)
else:
    # If already numeric-ish, treat nonzero as yes
    df["children_yes"] = (df["feature6"] != 0).astype(int)

# Affair frequency
affair = df["feature2"].astype(float)

# Create binary outcome: any affair (>0)
any_affair = (affair > 0).astype(int)
df["any_affair"] = any_affair

# Basic group stats
summary = df.groupby("children_yes")["feature2"].agg(["count", "mean", "median", "std"])
summary_any = df.groupby("children_yes")["any_affair"].agg(["mean", "count"])

# Two-sample t-test (unequal variance) for mean affair frequency
grp0 = affair[df["children_yes"] == 0]
grp1 = affair[df["children_yes"] == 1]

ttest = stats.ttest_ind(grp1, grp0, equal_var=False, nan_policy="omit")

# Mann-Whitney U test (nonparametric)
try:
    mwu = stats.mannwhitneyu(grp1, grp0, alternative="two-sided")
except ValueError:
    mwu = None

# Difference in proportion of any affair (two-proportion z-test)
# Using statsmodels proportions_ztest
from statsmodels.stats.proportion import proportions_ztest

count = np.array([
    df.loc[df["children_yes"] == 1, "any_affair"].sum(),
    df.loc[df["children_yes"] == 0, "any_affair"].sum(),
])
nobs = np.array([
    (df["children_yes"] == 1).sum(),
    (df["children_yes"] == 0).sum(),
])
prop_test = proportions_ztest(count, nobs)

# Logistic regression for any affair ~ children + controls
# Build formula with controls; use C() for categorical gender if needed
# feature3: gender category
formula_logit = "any_affair ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"

logit_model = smf.logit(formula_logit, data=df).fit(disp=False)

# OLS regression for affair frequency
formula_ols = "feature2 ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
ols_model = smf.ols(formula_ols, data=df).fit(cov_type="HC1")

# Extract key results
results = {
    "summary_mean": summary.to_dict(),
    "summary_any": summary_any.to_dict(),
    "ttest": {"stat": float(ttest.statistic), "pvalue": float(ttest.pvalue)},
    "mwu": None if mwu is None else {"stat": float(mwu.statistic), "pvalue": float(mwu.pvalue)},
    "prop_test": {"stat": float(prop_test[0]), "pvalue": float(prop_test[1])},
    "logit": {
        "coef_children": float(logit_model.params["children_yes"]),
        "p_children": float(logit_model.pvalues["children_yes"]),
        "odds_ratio_children": float(np.exp(logit_model.params["children_yes"]))
    },
    "ols": {
        "coef_children": float(ols_model.params["children_yes"]),
        "p_children": float(ols_model.pvalues["children_yes"])
    }
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
