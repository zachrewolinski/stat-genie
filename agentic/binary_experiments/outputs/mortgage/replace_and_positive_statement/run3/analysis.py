import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv("mortgage.csv")

# Basic cleaning: drop rows with missing values in used columns
features = [
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
used_cols = ["accept"] + features
_df = _df[used_cols].dropna()

# Create a binary proxy for female (values are noisy, so threshold at 0.5)
_df["female_bin"] = (_df["female"] >= 0.5).astype(int)

# Unadjusted acceptance rates by gender proxy
rates = _df.groupby("female_bin")["accept"].mean().rename({0: "male", 1: "female"})
print("Unadjusted acceptance rates (by female_bin):")
print(rates)
print("Unadjusted difference (female - male):", rates.get("female", float("nan")) - rates.get("male", float("nan")))

# Logistic regression with controls
X = _df[features]
X = sm.add_constant(X)
y = _df["accept"]

logit_model = sm.Logit(y, X)
logit_res = logit_model.fit(disp=False)
print("\nLogit coefficient for female:", logit_res.params["female"])
print("Logit p-value for female:", logit_res.pvalues["female"])

# Linear probability model with robust SEs as a check
ols_res = sm.OLS(y, X).fit(cov_type="HC3")
print("\nOLS coefficient for female:", ols_res.params["female"])
print("OLS p-value for female:", ols_res.pvalues["female"])

# Save key results to a small dictionary for quick inspection if needed
results = {
    "accept_rate_male": float(rates.get("male", float("nan"))),
    "accept_rate_female": float(rates.get("female", float("nan"))),
    "logit_female_coef": float(logit_res.params["female"]),
    "logit_female_p": float(logit_res.pvalues["female"]),
    "ols_female_coef": float(ols_res.params["female"]),
    "ols_female_p": float(ols_res.pvalues["female"]),
}

print("\nSummary:", results)
