import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Focus on variables relevant to mortgage applications
cols = [
    "female",
    "accept",
    "deny",
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

df = _df[cols].copy()

# Drop missing values in analysis columns
clean = df.dropna()

# Basic counts
female_counts = clean["female"].value_counts().to_dict()
accept_counts = clean["accept"].value_counts().to_dict()

# Approval rates by gender
approval_by_gender = clean.groupby("female")["accept"].mean()

# Chi-square test of independence between gender and approval
cont_table = pd.crosstab(clean["female"], clean["accept"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(cont_table)

# Logistic regression: accept ~ female + controls
# (use mortgage-related controls to isolate gender effect)
X = clean[[
    "female",
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
]]

# Add intercept
X = sm.add_constant(X, has_constant="add")
y = clean["accept"]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

coef_female = result.params["female"]
se_female = result.bse["female"]
p_female = result.pvalues["female"]

# Odds ratio and 95% CI for female
or_female = np.exp(coef_female)
ci_low = np.exp(coef_female - 1.96 * se_female)
ci_high = np.exp(coef_female + 1.96 * se_female)

# Output results in a simple, parseable format
print("N_clean", len(clean))
print("female_counts", female_counts)
print("accept_counts", accept_counts)
print("approval_by_gender", approval_by_gender.to_dict())
print("chi2", chi2)
print("chi2_p", p_chi2)
print("logit_coef_female", coef_female)
print("logit_p_female", p_female)
print("logit_or_female", or_female)
print("logit_or_ci", (ci_low, ci_high))
