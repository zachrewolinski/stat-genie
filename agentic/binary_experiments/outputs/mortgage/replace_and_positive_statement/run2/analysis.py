import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic checks
# Treat accept as outcome (1 = accepted). If accept missing, derive from deny
if "accept" not in df.columns and "deny" in df.columns:
    df["accept"] = 1 - df["deny"]

# Define features for model
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

# Drop rows with missing values in outcome or features
model_df = df[["accept"] + features].dropna().copy()

# Add intercept
X = sm.add_constant(model_df[features])
y = model_df["accept"]

# Fit logistic regression
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

# Also compute unadjusted acceptance rates by gender
# For robustness with non-binary values, treat female >= 0.5 as female
female_flag = (df["female"] >= 0.5).astype(int)
accept = df["accept"]
rate_female = accept[female_flag == 1].mean()
rate_male = accept[female_flag == 0].mean()

# Save key outputs to a small results dict for later use
summary_dict = {
    "n_obs": int(model_df.shape[0]),
    "female_coef": float(result.params["female"]),
    "female_pvalue": float(result.pvalues["female"]),
    "female_odds_ratio": float(np.exp(result.params["female"])),
    "rate_female": float(rate_female),
    "rate_male": float(rate_male),
}

print("Logit model fitted. Key results:")
for k, v in summary_dict.items():
    print(f"{k}: {v}")

# Save full model summary for inspection
with open("analysis_summary.txt", "w") as f:
    f.write(result.summary2().as_text())
    f.write("\n\nUnadjusted acceptance rates (female>=0.5):\n")
    f.write(f"female: {rate_female:.4f}\n")
    f.write(f"male: {rate_male:.4f}\n")

# Save a concise csv with coefficients
coef_df = pd.DataFrame({
    "coef": result.params,
    "pvalue": result.pvalues,
    "odds_ratio": np.exp(result.params),
})
coef_df.to_csv("analysis_coefficients.csv", index=True)
