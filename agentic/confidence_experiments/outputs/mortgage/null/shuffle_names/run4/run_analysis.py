import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Based on info.json metadata mapping
# gender: column 'denied_PMI' (1 female, 0 male)
# outcome: column 'self_employed' (1 denied, 0 accepted)

# Prepare analysis dataset
cols_needed = df.columns.tolist()

# Basic variables
gender_col = "denied_PMI"
outcome_col = "self_employed"  # 1=denied, 0=accepted

# Clean
analysis_df = df[[gender_col, outcome_col]].copy()
analysis_df = analysis_df.dropna()

# Contingency table
ct = pd.crosstab(analysis_df[gender_col], analysis_df[outcome_col])

# Approval rates
analysis_df["approved"] = 1 - analysis_df[outcome_col]
approval_rates = analysis_df.groupby(gender_col)["approved"].mean()

# Two-proportion z-test for approval rate difference
# female=1, male=0
n_female = (analysis_df[gender_col] == 1).sum()
n_male = (analysis_df[gender_col] == 0).sum()

approved_female = analysis_df.loc[analysis_df[gender_col] == 1, "approved"].sum()
approved_male = analysis_df.loc[analysis_df[gender_col] == 0, "approved"].sum()

# Use proportions_ztest from statsmodels if available, otherwise compute manually
from statsmodels.stats.proportion import proportions_ztest

count = np.array([approved_female, approved_male])
obs = np.array([n_female, n_male])
stat, pval = proportions_ztest(count, obs)

# Difference in approval rates
rate_female = approved_female / n_female if n_female > 0 else np.nan
rate_male = approved_male / n_male if n_male > 0 else np.nan
rate_diff = rate_female - rate_male

# 95% CI for difference in proportions
# standard error for difference
p1 = rate_female
p2 = rate_male
se = np.sqrt(p1*(1-p1)/n_female + p2*(1-p2)/n_male)
ci_low = rate_diff - 1.96 * se
ci_high = rate_diff + 1.96 * se

# Logistic regression with controls
# Exclude obvious ID and post-decision outcome columns
exclude_cols = {"bad_history", gender_col, outcome_col, "deny"}
# We also exclude 'female' column since it's a different binary outcome (PMI) per metadata
exclude_cols.add("female")

control_cols = [c for c in df.columns if c not in exclude_cols]

# Build model data
model_df = df[[outcome_col, gender_col] + control_cols].copy()
model_df = model_df.dropna()

# add constant
X = model_df[[gender_col] + control_cols]
X = sm.add_constant(X)
y = model_df[outcome_col]

logit_model = sm.Logit(y, X)
try:
    result = logit_model.fit(disp=False)
    coef = result.params[gender_col]
    p_value = result.pvalues[gender_col]
    # odds ratio
    odds_ratio = float(np.exp(coef))
except Exception as e:
    result = None
    coef = np.nan
    p_value = np.nan
    odds_ratio = np.nan
    print("Logit failed", e)

# Output summary stats
print("n", len(analysis_df))
print("female count", n_female, "male count", n_male)
print("approval rates female", rate_female, "male", rate_male)
print("diff", rate_diff, "95% CI", (ci_low, ci_high))
print("ztest p", pval)

if result is not None:
    print("logit coef female", coef, "odds ratio", odds_ratio, "p", p_value)
    print("logit n", result.nobs)
