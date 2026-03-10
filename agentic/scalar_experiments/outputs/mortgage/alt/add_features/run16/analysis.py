import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic checks
# Ensure binary columns are numeric

# Define variables
outcome = "deny"  # 1 denied, 0 accepted
exposure = "female"  # 1 female, 0 male

# Drop rows with missing values in outcome/exposure
base = df[[outcome, exposure]].dropna()

# Compute denial rates by gender
rates = base.groupby(exposure)[outcome].mean()
counts = base.groupby(exposure)[outcome].count()

# Two-proportion z-test for difference in denial rates
count_denied = base.groupby(exposure)[outcome].sum().values
nobs = base.groupby(exposure)[outcome].count().values
# Ensure order [0,1] (male, female)
order = [0,1]
count_denied = base.groupby(exposure)[outcome].sum().reindex(order).values
nobs = base.groupby(exposure)[outcome].count().reindex(order).values

zstat, pval = proportions_ztest(count_denied, nobs)

# Effect size: difference in denial rates
rate_male = rates.get(0, np.nan)
rate_female = rates.get(1, np.nan)
rate_diff = rate_female - rate_male

# Unadjusted logistic regression
X_unadj = sm.add_constant(base[exposure])
model_unadj = sm.Logit(base[outcome], X_unadj).fit(disp=False)

# Adjusted model with plausible credit risk covariates
covariates = [
    exposure,
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

# Keep rows with all covariates
adj = df[[outcome] + covariates].dropna()

X_adj = sm.add_constant(adj[covariates])
model_adj = sm.Logit(adj[outcome], X_adj).fit(disp=False)

# Extract female coefficient and p-value
coef_unadj = model_unadj.params[exposure]
se_unadj = model_unadj.bse[exposure]

coef_adj = model_adj.params[exposure]
se_adj = model_adj.bse[exposure]

# Alternative adjusted model excluding denied_PMI (potentially post-decision)
covariates_alt = [
    exposure,
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

adj_alt = df[[outcome] + covariates_alt].dropna()
X_adj_alt = sm.add_constant(adj_alt[covariates_alt])
model_adj_alt = sm.Logit(adj_alt[outcome], X_adj_alt).fit(disp=False)

coef_adj_alt = model_adj_alt.params[exposure]
se_adj_alt = model_adj_alt.bse[exposure]

# Odds ratios and 95% CI
or_unadj = np.exp(coef_unadj)
ci_unadj = np.exp(coef_unadj + np.array([-1, 1]) * 1.96 * se_unadj)

or_adj = np.exp(coef_adj)
ci_adj = np.exp(coef_adj + np.array([-1, 1]) * 1.96 * se_adj)

p_unadj = model_unadj.pvalues[exposure]
p_adj = model_adj.pvalues[exposure]

or_adj_alt = np.exp(coef_adj_alt)
ci_adj_alt = np.exp(coef_adj_alt + np.array([-1, 1]) * 1.96 * se_adj_alt)
p_adj_alt = model_adj_alt.pvalues[exposure]

# Save summary metrics to a json-ish dict for reporting
summary = {
    "n_total": int(len(df)),
    "n_base": int(len(base)),
    "n_adj": int(len(adj)),
    "denial_rate_male": float(rate_male),
    "denial_rate_female": float(rate_female),
    "denial_rate_diff_female_minus_male": float(rate_diff),
    "ztest_pvalue": float(pval),
    "logit_unadj_or": float(or_unadj),
    "logit_unadj_ci_low": float(ci_unadj[0]),
    "logit_unadj_ci_high": float(ci_unadj[1]),
    "logit_unadj_pvalue": float(p_unadj),
    "logit_adj_or": float(or_adj),
    "logit_adj_ci_low": float(ci_adj[0]),
    "logit_adj_ci_high": float(ci_adj[1]),
    "logit_adj_pvalue": float(p_adj),
    "n_adj_alt": int(len(adj_alt)),
    "logit_adj_alt_or": float(or_adj_alt),
    "logit_adj_alt_ci_low": float(ci_adj_alt[0]),
    "logit_adj_alt_ci_high": float(ci_adj_alt[1]),
    "logit_adj_alt_pvalue": float(p_adj_alt),
}

print(summary)
