import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import Table2x2

# Load data

df = pd.read_csv("mortgage.csv")

# Clean/ensure types
# Use female as 1=female, 0=male; accept as 1 accepted

# Basic counts
n_total = len(df)

# Drop rows with missing relevant fields
subset_cols = [
    "female",
    "accept",
    "deny",
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

df_sub = df[subset_cols].dropna()

n_used = len(df_sub)

# Approval rates by gender
rate_female = df_sub.loc[df_sub["female"] == 1, "accept"].mean()
rate_male = df_sub.loc[df_sub["female"] == 0, "accept"].mean()

n_female = (df_sub["female"] == 1).sum()
n_male = (df_sub["female"] == 0).sum()

# Two-proportion z-test for difference in acceptance rates
successes = np.array([
    df_sub.loc[df_sub["female"] == 1, "accept"].sum(),
    df_sub.loc[df_sub["female"] == 0, "accept"].sum(),
])
ns = np.array([n_female, n_male])

zstat, pval = proportions_ztest(successes, ns)

# 2x2 contingency for odds ratio (unadjusted)
# rows: female=1, female=0; cols: accept=1, accept=0
ct = pd.crosstab(df_sub["female"], df_sub["accept"]).reindex(index=[1,0], columns=[1,0])
# Ensure no zero cells for OR computation; use Table2x2 (adds no correction)

ct_values = ct.values
# ct order: [[female accept, female deny], [male accept, male deny]]
# But with columns [1,0], yes.

table = Table2x2(ct_values)
unadj_or = table.oddsratio
unadj_or_ci = table.oddsratio_confint()

# Logistic regression with controls
X = df_sub[[
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
]]

X = sm.add_constant(X, has_constant="add")

y = df_sub["accept"]

logit = sm.Logit(y, X)
res = logit.fit(disp=False)

# Robust SE (HC1) applied in-place
res._get_robustcov_results(cov_type="HC1")
coef_female = res.params["female"]
se_female = res.bse["female"]

# Wald p-value (robust)
p_female = res.pvalues["female"]

# Odds ratio and CI for female
or_female = float(np.exp(coef_female))
ci_low = float(np.exp(coef_female - 1.96 * se_female))
ci_high = float(np.exp(coef_female + 1.96 * se_female))

results = {
    "n_total": int(n_total),
    "n_used": int(n_used),
    "n_female": int(n_female),
    "n_male": int(n_male),
    "rate_female": float(rate_female),
    "rate_male": float(rate_male),
    "rate_diff_female_minus_male": float(rate_female - rate_male),
    "z_test_p": float(pval),
    "z_test_z": float(zstat),
    "unadjusted_or": float(unadj_or),
    "unadjusted_or_ci_low": float(unadj_or_ci[0]),
    "unadjusted_or_ci_high": float(unadj_or_ci[1]),
    "logit_coef_female": float(coef_female),
    "logit_or_female": float(or_female),
    "logit_or_ci_low": float(ci_low),
    "logit_or_ci_high": float(ci_high),
    "logit_p_female": float(p_female),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
