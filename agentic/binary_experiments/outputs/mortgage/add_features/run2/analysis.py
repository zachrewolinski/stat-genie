import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Outcome: 1 if denied
outcome = "deny"

# Core covariates related to creditworthiness + demographics
covariates = [
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
    "occupation",
    "age",
]

# Keep only needed columns and drop missing
cols = [outcome] + covariates
model_df = df[cols].dropna().copy()

# Basic summary stats
n_total = len(model_df)
rate_by_gender = model_df.groupby("female")[outcome].mean()
rate_diff = rate_by_gender.get(1.0, np.nan) - rate_by_gender.get(0.0, np.nan)

# Logistic regression with robust (HC1) SE
X = sm.add_constant(model_df[covariates])
y = model_df[outcome]
logit = sm.Logit(y, X)
res = logit.fit(disp=False, cov_type="HC1")

female_coef = res.params.get("female", np.nan)
female_se = res.bse.get("female", np.nan)
female_p = res.pvalues.get("female", np.nan)

# Odds ratio for female
female_or = np.exp(female_coef) if pd.notna(female_coef) else np.nan

# Marginal effect for female (average, if available)
try:
    meff = res.get_margeff(at="overall", method="dydx")
    meff_df = meff.summary_frame()
    female_me = float(meff_df.loc["female", "dy/dx"]) if "female" in meff_df.index else np.nan
except Exception:
    female_me = np.nan

print("N used:", n_total)
print("Denial rate by gender (female=1, male=0):")
print(rate_by_gender)
print("Unadjusted denial rate difference (female - male):", rate_diff)
print("Logit female coef (robust SE):", female_coef, "SE:", female_se, "p:", female_p)
print("Female odds ratio:", female_or)
print("Female marginal effect (overall):", female_me)
