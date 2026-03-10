import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

print("rows", len(df))
print("columns", df.columns.tolist())

# Basic sanity: check accept/deny relation
if "accept" in df.columns and "deny" in df.columns:
    mismatch = (df["accept"] != (1 - df["deny"])) & df[["accept", "deny"]].notna().all(axis=1)
    print("accept_deny_mismatch_count", int(mismatch.sum()))

# Focus variables
key_cols = [
    "accept",
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

missing = df[key_cols].isna().sum()
print("missing_key_cols", missing.to_dict())

# Drop rows with missing in key columns
model_df = df[key_cols].dropna().copy()

# Acceptance rates by gender
rates = model_df.groupby("female")["accept"].agg(["mean", "count"])
print("accept_rate_by_female", rates)

# Difference in proportions test (female vs male)
# female=1, male=0
counts = model_df.groupby("female")["accept"].sum().astype(int)
ns = model_df.groupby("female")["accept"].count().astype(int)

if set(counts.index) == {0.0, 1.0} or set(counts.index) == {0, 1}:
    # order: female=1 vs female=0
    count = np.array([counts.loc[1], counts.loc[0]])
    nobs = np.array([ns.loc[1], ns.loc[0]])
    stat, pval = proportions_ztest(count, nobs)
    # difference in proportions
    p1 = counts.loc[1] / ns.loc[1]
    p0 = counts.loc[0] / ns.loc[0]
    diff = p1 - p0
    # 95% CI for difference (Wald)
    se = np.sqrt(p1 * (1 - p1) / ns.loc[1] + p0 * (1 - p0) / ns.loc[0])
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    print("proportion_test_stat", stat)
    print("proportion_test_pvalue", pval)
    print("accept_rate_diff_female_minus_male", diff)
    print("accept_rate_diff_ci", (ci_low, ci_high))
else:
    print("unexpected_female_values", counts.index.tolist())

# Logistic regression controlling for applicant characteristics
X_cols = [
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

X = model_df[X_cols]
X = sm.add_constant(X, has_constant="add")
y = model_df["accept"]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

print(result.summary())

# Extract female coefficient and odds ratio with CI
params = result.params
conf = result.conf_int()

female_coef = params.get("female", np.nan)
female_p = result.pvalues.get("female", np.nan)

if "female" in conf.index:
    ci_low, ci_high = conf.loc["female"].tolist()
    # odds ratio and CI
    or_female = np.exp(female_coef)
    or_low = np.exp(ci_low)
    or_high = np.exp(ci_high)
    print("female_logit_coef", female_coef)
    print("female_logit_pvalue", female_p)
    print("female_odds_ratio", or_female)
    print("female_or_ci", (or_low, or_high))

# Also check unadjusted logit with only female
X2 = sm.add_constant(model_df[["female"]], has_constant="add")
logit_model2 = sm.Logit(y, X2)
result2 = logit_model2.fit(disp=False)
print(result2.summary())
print("female_unadjusted_coef", result2.params.get("female", np.nan))
print("female_unadjusted_pvalue", result2.pvalues.get("female", np.nan))

