import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv("mortgage.csv")

# Basic cleaning
if "Unnamed: 0" in _df.columns:
    _df = _df.drop(columns=["Unnamed: 0"])

# Outcome: deny (1 = denied)
# Primary predictor: female (1 = female)

# Unadjusted denial rates
rates = _df.groupby("female")["deny"].mean()
count = _df.groupby("female")["deny"].count()

# Logistic regression with controls
controls = [
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

X = _df[controls].copy()
y = _df["deny"].astype(float)

# Drop rows with missing/inf values in model inputs
X = X.replace([np.inf, -np.inf], np.nan)
model_df = pd.concat([y, X], axis=1).dropna()
y = model_df["deny"]
X = model_df[controls]
X = sm.add_constant(X, has_constant="add")

model = sm.Logit(y, X)
result = model.fit(disp=False)

female_coef = result.params.get("female")
female_p = result.pvalues.get("female")

# Print key outputs for inspection
print("Unadjusted denial rates by female (0=male, 1=female):")
for k in rates.index:
    print(f"  female={int(k)}: rate={rates.loc[k]:.4f} (n={int(count.loc[k])})")

print("\nLogit deny ~ female + controls:")
print(f"  female coef (log-odds): {female_coef:.6f}")
print(f"  female p-value: {female_p:.6g}")

# Also compute odds ratio for interpretability
odds_ratio = float(np.exp(female_coef)) if female_coef is not None else np.nan
print(f"  female odds ratio: {odds_ratio:.4f}")
