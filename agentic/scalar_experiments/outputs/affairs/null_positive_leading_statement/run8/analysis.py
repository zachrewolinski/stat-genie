import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv("affairs.csv")

# Basic variables
_df["affairs_any"] = (_df["affairs"] > 0).astype(int)
_df["children_yes"] = (_df["children"].str.lower() == "yes").astype(int)

# Group summaries
summary = _df.groupby("children_yes").agg(
    n=("affairs", "size"),
    mean_affairs=("affairs", "mean"),
    median_affairs=("affairs", "median"),
    any_affair_rate=("affairs_any", "mean"),
)

# T-test for mean affairs (Welch)
with_children = _df.loc[_df["children_yes"] == 1, "affairs"]
no_children = _df.loc[_df["children_yes"] == 0, "affairs"]

ttest = stats.ttest_ind(with_children, no_children, equal_var=False)

# Logistic regression for any affair controlling for covariates
# Use categorical encoding for gender
logit_formula = (
    "affairs_any ~ children_yes + C(gender) + age + yearsmarried + "
    "religiousness + education + occupation + rating"
)
logit_model = smf.logit(logit_formula, data=_df).fit(disp=0)

# OLS on log(affairs+1) controlling for covariates
_df["log_affairs"] = np.log1p(_df["affairs"])
ols_formula = (
    "log_affairs ~ children_yes + C(gender) + age + yearsmarried + "
    "religiousness + education + occupation + rating"
)
ols_model = smf.ols(ols_formula, data=_df).fit()

# Extract key effects
logit_coef = logit_model.params["children_yes"]
logit_p = logit_model.pvalues["children_yes"]
logit_or = float(np.exp(logit_coef))

ols_coef = ols_model.params["children_yes"]
ols_p = ols_model.pvalues["children_yes"]

# Determine evidence of decrease
# Decrease means children_yes is associated with lower affairs => negative coef and OR < 1
# Use a simple score based on consistency and significance
score = 0.0

# Unadjusted difference in mean
mean_diff = summary.loc[1, "mean_affairs"] - summary.loc[0, "mean_affairs"]
if mean_diff < 0:
    score += 0.25

# t-test significance
if ttest.pvalue < 0.05 and mean_diff < 0:
    score += 0.15

# Logistic regression
if logit_coef < 0:
    score += 0.30
if logit_p < 0.05 and logit_coef < 0:
    score += 0.20

# OLS regression
if ols_coef < 0:
    score += 0.05
if ols_p < 0.05 and ols_coef < 0:
    score += 0.05

# Cap score between 0 and 1
score = max(0.0, min(1.0, score))

# Map score to Likert scale for "Yes" (decrease)
# 0 -> 0, 1 -> 100
likert = int(round(score * 100))

# If evidence suggests increase (opposite), push negative based on magnitude
if (logit_coef > 0 and ols_coef > 0 and mean_diff > 0):
    # reverse score for strong no
    likert = -int(round(score * 100))

# Save results for inspection
summary.to_csv("analysis_summary.csv")

with open("analysis_results.txt", "w") as f:
    f.write("Summary by children:\n")
    f.write(summary.to_string())
    f.write("\n\nT-test (Welch) for mean affairs (with_children - no_children):\n")
    f.write(f"mean_diff = {mean_diff:.4f}, t = {ttest.statistic:.4f}, p = {ttest.pvalue:.6f}\n")
    f.write("\nLogit (any affair) children_yes effect:\n")
    f.write(f"coef = {logit_coef:.4f}, OR = {logit_or:.4f}, p = {logit_p:.6f}\n")
    f.write("\nOLS (log1p affairs) children_yes effect:\n")
    f.write(f"coef = {ols_coef:.4f}, p = {ols_p:.6f}\n")
    f.write(f"\nLikert score (Yes=decrease): {likert}\n")

with open("conclusion.txt", "w") as f:
    f.write(str(likert))

print(likert)
