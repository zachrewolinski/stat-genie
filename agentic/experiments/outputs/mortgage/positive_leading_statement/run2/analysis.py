import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv("mortgage.csv")

# Basic cleaning: drop index-like column if present
if "Unnamed: 0" in _df.columns:
    _df = _df.drop(columns=["Unnamed: 0"])

# Ensure binary outcomes are numeric
_df["accept"] = pd.to_numeric(_df["accept"], errors="coerce")
_df["female"] = pd.to_numeric(_df["female"], errors="coerce")

# Drop rows with missing key fields
_df = _df.dropna(subset=["accept", "female"])

# Unadjusted approval rates by gender
rates = _df.groupby("female")["accept"].mean()
counts = _df.groupby("female")["accept"].agg(["count", "sum"])

# Chi-square test of independence
contingency = pd.crosstab(_df["female"], _df["accept"])
chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)

# Adjusted logistic regression
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
]

# Keep only available columns and drop missing
cols = [c for c in covariates if c in _df.columns]
model_df = _df.dropna(subset=cols + ["accept"]).copy()

X = model_df[cols]
X = sm.add_constant(X, has_constant="add")
y = model_df["accept"]

logit = sm.Logit(y, X)
result = logit.fit(disp=False)

female_coef = result.params.get("female", np.nan)
female_p = result.pvalues.get("female", np.nan)

# Convert coefficient to odds ratio for interpretability
female_or = np.exp(female_coef) if np.isfinite(female_coef) else np.nan

print("Unadjusted approval rates (accept rate):")
print(rates)
print("Counts (accept sums):")
print(counts)
print("Chi-square p-value:", p_chi2)
print("Adjusted logit female coef:", female_coef)
print("Adjusted logit female odds ratio:", female_or)
print("Adjusted logit female p-value:", female_p)
