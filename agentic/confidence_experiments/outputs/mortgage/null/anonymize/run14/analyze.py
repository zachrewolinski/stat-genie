import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
DATA_PATH = "mortgage.csv"
df = pd.read_csv(DATA_PATH)

# Replace inf with nan, then drop rows with any missing values in relevant columns
df = df.replace([np.inf, -np.inf], np.nan)

# Define variables
# Outcome: approval (feature14 == 1)
outcome = "feature14"  # accepted

# Gender variable: 1 if female, 0 if male
female = "feature2"

# Basic sanity check: feature11 should be denial (1 if denied)
# We'll use feature14 as outcome and exclude feature11 to avoid redundancy.

# Compute approval rates by gender
rates = df.groupby(female)[outcome].agg(["mean", "count", "sum"]).rename(columns={"mean": "approval_rate", "sum": "approved"})

# Two-proportion z-test (female vs male)
# female=1 vs male=0
female_group = df[df[female] == 1][outcome]
male_group = df[df[female] == 0][outcome]

n1 = female_group.shape[0]
n0 = male_group.shape[0]

p1 = female_group.mean()
p0 = male_group.mean()

# Pooled proportion
p_pool = (female_group.sum() + male_group.sum()) / (n1 + n0)

# Standard error and z-test
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
if se == 0:
    z = np.nan
    p_value = np.nan
else:
    z = (p1 - p0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

# Logistic regression with controls
# Use all features except outcome and denial feature11 as predictors
predictors = [col for col in df.columns if col not in [outcome, "feature11"]]
# Drop rows with any missing values in predictors or outcome
model_df = df[predictors + [outcome]].dropna()

X = model_df[predictors]
X = sm.add_constant(X, has_constant="add")
y = model_df[outcome]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

# Extract gender coefficient
coef = result.params[female]
se_coef = result.bse[female]
pval_coef = result.pvalues[female]

# Convert to odds ratio
odds_ratio = np.exp(coef)

# Also compute marginal effect at means (approx) for interpretability
# Use statsmodels to get marginal effects
try:
    marg = result.get_margeff(at="mean", method="dydx")
    marg_eff = float(marg.margeff[list(marg.params.index).index(female)])
    marg_pval = float(marg.pvalues[list(marg.params.index).index(female)])
except Exception:
    marg_eff = np.nan
    marg_pval = np.nan

# Collect results
summary = {
    "approval_rates": rates.to_dict(orient="index"),
    "two_proportion_test": {
        "female_rate": float(p1),
        "male_rate": float(p0),
        "z": float(z) if z == z else None,
        "p_value": float(p_value) if p_value == p_value else None,
    },
    "logit_gender": {
        "coef": float(coef),
        "se": float(se_coef),
        "p_value": float(pval_coef),
        "odds_ratio": float(odds_ratio),
        "marginal_effect": float(marg_eff) if marg_eff == marg_eff else None,
        "marginal_p_value": float(marg_pval) if marg_pval == marg_pval else None,
    },
    "n": int(df.shape[0]),
    "n_logit": int(model_df.shape[0]),
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
