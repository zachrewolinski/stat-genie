import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleanup
# Ensure binary columns are numeric 0/1
binary_cols = ["female", "black", "self_employed", "married", "bad_history", "denied_PMI", "accept", "deny"]
for col in binary_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing key variables
key_cols = [
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
    "denied_PMI",
]

df_clean = df.dropna(subset=key_cols).copy()

# Descriptive stats
n_total = len(df_clean)
accept_rate_overall = df_clean["accept"].mean()

accept_rate_female = df_clean.loc[df_clean["female"] == 1, "accept"].mean()
accept_rate_male = df_clean.loc[df_clean["female"] == 0, "accept"].mean()

n_female = (df_clean["female"] == 1).sum()
n_male = (df_clean["female"] == 0).sum()

# Difference in proportions and CI
# Wald CI for difference in proportions
p1 = accept_rate_female
p0 = accept_rate_male
se_diff = np.sqrt(p1*(1-p1)/n_female + p0*(1-p0)/n_male)
diff = p1 - p0
z = stats.norm.ppf(0.975)
ci_low = diff - z*se_diff
ci_high = diff + z*se_diff

# Chi-square test of independence
contingency = pd.crosstab(df_clean["female"], df_clean["accept"])
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression: unadjusted
X1 = sm.add_constant(df_clean[["female"]])
model1 = sm.Logit(df_clean["accept"], X1)
res1 = model1.fit(disp=False)

# Logistic regression: adjusted
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
X2 = sm.add_constant(df_clean[predictors])
model2 = sm.Logit(df_clean["accept"], X2)
res2 = model2.fit(disp=False)

# Extract female effect
coef1 = res1.params["female"]
se1 = res1.bse["female"]
pval1 = res1.pvalues["female"]
or1 = np.exp(coef1)
ci1 = np.exp(res1.conf_int().loc["female"].values)

coef2 = res2.params["female"]
se2 = res2.bse["female"]
pval2 = res2.pvalues["female"]
or2 = np.exp(coef2)
ci2 = np.exp(res2.conf_int().loc["female"].values)

# Output summary for reporting
print("N total:", n_total)
print("Accept rate overall:", accept_rate_overall)
print("Accept rate female:", accept_rate_female)
print("Accept rate male:", accept_rate_male)
print("Difference (female - male):", diff)
print("95% CI diff:", (ci_low, ci_high))
print("Chi-square p-value:", p_chi2)
print("\nUnadjusted logit female coef:", coef1, "SE:", se1, "p:", pval1)
print("Unadjusted OR:", or1, "95% CI:", ci1)
print("\nAdjusted logit female coef:", coef2, "SE:", se2, "p:", pval2)
print("Adjusted OR:", or2, "95% CI:", ci2)

print("\nNote: Robust (HC3) SEs not computed for Logit in this environment.")
