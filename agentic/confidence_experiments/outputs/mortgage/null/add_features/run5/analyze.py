import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Relevant columns for mortgage analysis
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
    "denied_PMI",
]

# Keep only needed columns and drop missing
use_df = df[cols].copy()
use_df = use_df.dropna()

# Ensure numeric
for c in cols:
    use_df[c] = pd.to_numeric(use_df[c], errors="coerce")
use_df = use_df.dropna()

# Basic counts
n_total = len(use_df)

# Unadjusted approval rates by gender
rate_by_gender = use_df.groupby("female")["accept"].mean()
count_by_gender = use_df.groupby("female")["accept"].count()

# Two-proportion z-test (female=1 vs male=0)
count_accept = use_df.groupby("female")["accept"].sum()
# order: female=1, female=0
n_female = int(count_by_gender.get(1, 0))
n_male = int(count_by_gender.get(0, 0))
accept_female = int(count_accept.get(1, 0))
accept_male = int(count_accept.get(0, 0))

z_stat = np.nan
p_value = np.nan
if n_female > 0 and n_male > 0:
    z_stat, p_value = proportions_ztest([accept_female, accept_male], [n_female, n_male])

# Logistic regression with controls
X = use_df[[
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
X = sm.add_constant(X)
y = use_df["accept"]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

coef_female = result.params.get("female", np.nan)
se_female = result.bse.get("female", np.nan)
p_female = result.pvalues.get("female", np.nan)

odds_ratio = np.exp(coef_female) if pd.notnull(coef_female) else np.nan

# 95% CI for odds ratio
if pd.notnull(coef_female) and pd.notnull(se_female):
    ci_low = coef_female - 1.96 * se_female
    ci_high = coef_female + 1.96 * se_female
    or_low = np.exp(ci_low)
    or_high = np.exp(ci_high)
else:
    or_low = np.nan
    or_high = np.nan

# Output summary to stdout for manual review
print("n_total", n_total)
print("rate_by_gender", rate_by_gender.to_dict())
print("count_by_gender", count_by_gender.to_dict())
print("accept_counts", {"female": accept_female, "male": accept_male})
print("z_stat", z_stat, "p_value", p_value)
print("logit_female_coef", coef_female, "se", se_female, "p", p_female)
print("odds_ratio", odds_ratio, "OR_95%", (or_low, or_high))
