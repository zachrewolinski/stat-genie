import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic cleanup
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Ensure numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with any missing values in used columns
outcome = "deny"
main_var = "female"
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
    "denied_PMI",
]
use_cols = [outcome, main_var] + controls

df_model = df[use_cols].dropna().copy()

# Logistic regression with and without controls
X_simple = sm.add_constant(df_model[[main_var]])
y = df_model[outcome]
logit_simple = sm.Logit(y, X_simple).fit(disp=False)

# Logistic regression with controls
X = df_model[[main_var] + controls]
X = sm.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

coef_female = result.params[main_var]
se_female = result.bse[main_var]
pval_female = result.pvalues[main_var]

# Convert to odds ratio
odds_ratio = float(np.exp(coef_female))

# Save summary for debugging (optional)
summary_text = result.summary2().as_text()
with open("analysis_results.txt", "w") as f:
    f.write("Logit (deny ~ female) results:\n")
    f.write(logit_simple.summary2().as_text())
    f.write("\n\nLogit (deny ~ female + controls) results:\n")
    f.write("\nLogit coefficient (female): {:.6f}\n".format(coef_female))
    f.write("Std err: {:.6f}\n".format(se_female))
    f.write("P-value: {:.6f}\n".format(pval_female))
    f.write("Odds ratio: {:.6f}\n\n".format(odds_ratio))
    f.write(summary_text)

# Write conclusion
# Decision rule: if p-value for female in multivariate logit < 0.05 -> "Yes"
# otherwise "No". Interpret effect direction.
if pval_female < 0.05:
    decision = "Yes"
    direction = "higher" if coef_female > 0 else "lower"
    reasoning = (
        f"In a logistic regression controlling for credit and application factors, "
        f"female has a statistically significant association with denial (p={pval_female:.3f}). "
        f"The estimated odds of denial are {direction} for female applicants (odds ratio {odds_ratio:.2f})."
    )
else:
    decision = "No"
    reasoning = (
        f"In a logistic regression controlling for credit and application factors, "
        f"female is not a statistically significant predictor of denial (p={pval_female:.3f})."
    )

with open("conclusion.txt", "w") as f:
    f.write(decision + "\n")
    f.write(reasoning + "\n")

print("Done. See analysis_results.txt and conclusion.txt")
