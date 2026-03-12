import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Use accept as approval outcome (1=accepted, 0=denied)
# Ensure numeric and drop rows with missing values in used columns
outcome = "accept"

# Basic columns
predictor = "female"

# Define control variables (exclude outcome and index-like column)
controls = [
    "black",
    "housing_expense_ratio",
    "self_employed",
    "married",
    "mortgage_credit",
    "consumer_credit",
    "bad_history",
    "PI_ratio",
    "loan_to_value",
    "denied_PMI",
]

# Clean dataset for analyses
analysis_cols = [outcome, predictor] + controls

df_clean = df[analysis_cols].dropna().copy()

# Compute acceptance rates by gender
rates = (
    df_clean.groupby(predictor)[outcome]
    .agg(["mean", "count", "sum"])
    .rename(index={0: "male", 1: "female"})
)

# Two-proportion z-test for difference in approval rates
# female=1, male=0
female_mask = df_clean[predictor] == 1
male_mask = df_clean[predictor] == 0

n_f = female_mask.sum()
n_m = male_mask.sum()

p_f = df_clean.loc[female_mask, outcome].mean()
p_m = df_clean.loc[male_mask, outcome].mean()

# pooled proportion
p_pool = (
    df_clean.loc[female_mask, outcome].sum() + df_clean.loc[male_mask, outcome].sum()
) / (n_f + n_m)

# z-test
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_f + 1 / n_m))
if se > 0:
    z = (p_f - p_m) / se
    p_value_diff = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_value_diff = np.nan

# Unadjusted logistic regression: accept ~ female
X_unadj = sm.add_constant(df_clean[[predictor]])
model_unadj = sm.Logit(df_clean[outcome], X_unadj).fit(disp=False)

# Adjusted logistic regression: accept ~ female + controls
X_adj = sm.add_constant(df_clean[[predictor] + controls])
model_adj = sm.Logit(df_clean[outcome], X_adj).fit(disp=False)

# Extract female coefficient stats
coef_unadj = model_unadj.params[predictor]
se_unadj = model_unadj.bse[predictor]
p_unadj = model_unadj.pvalues[predictor]

coef_adj = model_adj.params[predictor]
se_adj = model_adj.bse[predictor]
p_adj = model_adj.pvalues[predictor]

# Convert to odds ratios with 95% CI
or_unadj = float(np.exp(coef_unadj))
ci_unadj = np.exp(coef_unadj + np.array([-1, 1]) * 1.96 * se_unadj)

or_adj = float(np.exp(coef_adj))
ci_adj = np.exp(coef_adj + np.array([-1, 1]) * 1.96 * se_adj)

# Save results to JSON for use in report
results = {
    "n_total": int(len(df_clean)),
    "n_female": int(n_f),
    "n_male": int(n_m),
    "approval_rate_female": float(p_f),
    "approval_rate_male": float(p_m),
    "approval_rate_diff": float(p_f - p_m),
    "diff_z": float(z),
    "diff_p_value": float(p_value_diff),
    "unadjusted": {
        "coef": float(coef_unadj),
        "p_value": float(p_unadj),
        "odds_ratio": float(or_unadj),
        "ci_low": float(ci_unadj[0]),
        "ci_high": float(ci_unadj[1]),
    },
    "adjusted": {
        "coef": float(coef_adj),
        "p_value": float(p_adj),
        "odds_ratio": float(or_adj),
        "ci_low": float(ci_adj[0]),
        "ci_high": float(ci_adj[1]),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
