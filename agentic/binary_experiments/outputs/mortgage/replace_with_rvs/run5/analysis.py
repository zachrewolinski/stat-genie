import pandas as pd
import statsmodels.api as sm

# Load data
DF_PATH = "mortgage.csv"
df = pd.read_csv(DF_PATH)

# Basic cleanup
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Outcome: acceptance (1 accepted, 0 denied)
if "accept" in df.columns:
    y = df["accept"]
elif "deny" in df.columns:
    # If only deny exists, invert it
    y = 1 - df["deny"]
else:
    raise ValueError("No acceptance/denial outcome column found.")

# Predictor of interest
if "female" not in df.columns:
    raise ValueError("No 'female' column found for gender indicator.")

# Simple logistic regression: accept ~ female
X_simple = sm.add_constant(df[["female"]])
model_simple = sm.Logit(y, X_simple).fit(disp=0)

# Multivariate logistic regression with available controls
control_vars = [
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
controls_present = [c for c in control_vars if c in df.columns]
X_full = sm.add_constant(df[controls_present])
model_full = sm.Logit(y, X_full).fit(disp=0)

# Extract key stats
coef_simple = model_simple.params["female"]
se_simple = model_simple.bse["female"]
p_simple = model_simple.pvalues["female"]

coef_full = model_full.params["female"]
se_full = model_full.bse["female"]
p_full = model_full.pvalues["female"]

print("Rows:", len(df))
print("Acceptance rate:", y.mean())
print("\nSimple logit (accept ~ female):")
print("  coef:", coef_simple)
print("  se:", se_simple)
print("  p-value:", p_simple)

print("\nFull logit (accept ~ female + controls):")
print("  coef:", coef_full)
print("  se:", se_full)
print("  p-value:", p_full)
