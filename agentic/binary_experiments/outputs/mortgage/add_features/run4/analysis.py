import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
DATA_PATH = "mortgage.csv"
df = pd.read_csv(DATA_PATH)

# Focus on variables relevant to mortgage approval and applicant characteristics
cols = [
    "female",
    "accept",
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

sub = df[cols].copy()
sub = sub.dropna()

# Basic approval rates by gender
rate_by_gender = sub.groupby("female")["accept"].mean()
count_by_gender = sub.groupby("female")["accept"].size()
accept_counts = sub.groupby("female")["accept"].sum()

print("Approval rate by gender (female=1):")
print(rate_by_gender)
print("Counts by gender:")
print(count_by_gender)

# Two-proportion z-test for difference in approval rates
count = np.array([accept_counts.loc[0.0], accept_counts.loc[1.0]])
obs = np.array([count_by_gender.loc[0.0], count_by_gender.loc[1.0]])
stat, pval = proportions_ztest(count, obs)
print("\nTwo-proportion z-test (female vs male) for approval rate difference:")
print(f"z-stat: {stat:.3f}, p-value: {pval:.4f}")

# Logistic regression with controls
X = sub[[
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
]]
X = sm.add_constant(X)
y = sub["accept"]

logit = sm.Logit(y, X).fit(disp=False)
# Use robust standard errors to be conservative; handle older statsmodels
try:
    logit_robust = logit.get_robustcov_results(cov_type="HC3")
except AttributeError:
    # Older statsmodels: robust covariances are set in-place
    logit_robust = logit
    logit_robust._get_robustcov_results(cov_type="HC3")

print("\nLogit regression (accept ~ female + controls), robust SEs:")
print(logit_robust.summary())

# Effect size for female
coef = logit_robust.params["female"]
p_value = logit_robust.pvalues["female"]
odds_ratio = float(np.exp(coef))

margeff = logit_robust.get_margeff(at="overall")
me_df = margeff.summary_frame()
me = me_df.loc["female", "dy/dx"]
p_cols = [c for c in me_df.columns if "Pr(" in c or "P>|" in c]
me_p = me_df.loc["female", p_cols[0]] if p_cols else float("nan")

print("\nFemale effect (adjusted):")
print(f"coef={coef:.4f}, odds_ratio={odds_ratio:.3f}, p-value={p_value:.4f}")
print(f"marginal_effect={me:.4f}, p-value={me_p:.4f}")
