import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('mortgage.csv')

# Basic cleaning
# Ensure binary columns are numeric
for col in ["female", "accept", "deny", "black", "self_employed", "married", "bad_history", "denied_PMI"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Outcome: accept (1=accepted, 0=denied)
if "accept" not in df.columns:
    # fallback if accept not available
    df["accept"] = 1 - df["deny"]

# Simple acceptance rates by gender
rate_by_gender = df.groupby('female')["accept"].mean()

# Controls for creditworthiness and application characteristics
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
]

cols = ["accept", "female"] + controls
model_df = df[cols].dropna()

# Unadjusted model: accept ~ female
X_simple = sm.add_constant(model_df[["female"]], has_constant='add')
y_simple = model_df["accept"]
simple_model = sm.Logit(y_simple, X_simple)
simple_result = simple_model.fit(disp=0)

X = model_df[["female"] + controls]
X = sm.add_constant(X, has_constant='add')
y = model_df["accept"]

# Logistic regression
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=0)

# Extract female effect
coef = result.params["female"]
se = result.bse["female"]
pval = result.pvalues["female"]

# Odds ratio and 95% CI
odds_ratio = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

print("Acceptance rate by gender (female=0 male, female=1 female):")
print(rate_by_gender)
print("\nLogit model (accept ~ female)")
print(f"Female coef: {simple_result.params['female']:.4f}")
print(f"Female p-value: {simple_result.pvalues['female']:.4g}")

print("\nLogit model (accept ~ female + controls)")
print(f"Female coef: {coef:.4f}")
print(f"Female p-value: {pval:.4g}")
print(f"Female odds ratio: {odds_ratio:.4f}")
print(f"95% CI for odds ratio: [{ci_low:.4f}, {ci_high:.4f}]")
