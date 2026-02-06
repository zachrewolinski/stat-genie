import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Drop index-like column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Outcome and predictors
outcome = "accept"
predictors = [
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

# Keep only required columns and drop missing
use_cols = [outcome] + predictors
analysis_df = df[use_cols].dropna()

X = analysis_df[predictors]
X = sm.add_constant(X, has_constant="add")
y = analysis_df[outcome]

result = None
try:
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
except Exception as e:
    # Fallback to GLM binomial if Logit has convergence issues
    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

print(result.summary())

# Extract key stats for the female coefficient
if "female" in result.params.index:
    coef = result.params["female"]
    pval = result.pvalues["female"]
    # Compute odds ratio for interpretability
    odds_ratio = float(np.exp(coef))
    print("\nFemale coefficient:")
    print(f"  coef = {coef:.6f}")
    print(f"  p-value = {pval:.6g}")
    print(f"  odds ratio = {odds_ratio:.6f}")
else:
    print("\nFemale coefficient not found in model output.")
