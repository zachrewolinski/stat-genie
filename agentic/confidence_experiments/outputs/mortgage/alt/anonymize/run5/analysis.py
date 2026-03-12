import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic checks
n_rows = len(df)

# Identify columns
female_col = "feature2"  # 1 if female
accepted_col = "feature14"  # 1 if accepted

denied_col = "feature11"  # 1 if denied

# Check acceptance/denial consistency
consistency = ((df[accepted_col] + df[denied_col]) == 1).mean()

# Drop rows with missing in key cols
key_cols = [female_col, accepted_col]
base = df.dropna(subset=key_cols).copy()

# Unadjusted rates
rate_table = base.groupby(female_col)[accepted_col].agg(['mean','count'])
# mean is acceptance rate

# Chi-square test of independence on acceptance counts by gender
ct = pd.crosstab(base[female_col], base[accepted_col])
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Difference in proportions (female - male) and 95% CI
# 0: male, 1: female
male = base[base[female_col]==0]
female = base[base[female_col]==1]

p_m = male[accepted_col].mean()
p_f = female[accepted_col].mean()

n_m = len(male)
n_f = len(female)

# standard error for difference in proportions
se_diff = np.sqrt(p_m*(1-p_m)/n_m + p_f*(1-p_f)/n_f)

diff = p_f - p_m
z = stats.norm.ppf(0.975)
ci_low = diff - z*se_diff
ci_high = diff + z*se_diff

# Logistic regression unadjusted
base['intercept'] = 1
logit_unadj = sm.Logit(base[accepted_col], base[["intercept", female_col]]).fit(disp=False)

# Logistic regression adjusted for other covariates
# Use all other features except accepted/denied (and intercept added)
feature_cols = [c for c in df.columns if c not in [accepted_col, denied_col]]
# Ensure female is included

adj = df.dropna(subset=feature_cols + [accepted_col]).copy()
X = adj[feature_cols]
X = sm.add_constant(X, has_constant='add')
logit_adj = sm.Logit(adj[accepted_col], X).fit(disp=False)

# Extract female coefficient
coef_unadj = logit_unadj.params[female_col]
se_unadj = logit_unadj.bse[female_col]

coef_adj = logit_adj.params[female_col]
se_adj = logit_adj.bse[female_col]

# Odds ratios and 95% CIs
or_unadj = np.exp(coef_unadj)
or_adj = np.exp(coef_adj)

ci_unadj = np.exp(coef_unadj + np.array([-1,1]) * z * se_unadj)
ci_adj = np.exp(coef_adj + np.array([-1,1]) * z * se_adj)

p_unadj = logit_unadj.pvalues[female_col]
p_adj = logit_adj.pvalues[female_col]

summary = {
    "n_rows": int(n_rows),
    "consistency_accept_denied_rate": float(consistency),
    "accept_rate_male": float(p_m),
    "accept_rate_female": float(p_f),
    "n_male": int(n_m),
    "n_female": int(n_f),
    "diff_female_minus_male": float(diff),
    "diff_ci_low": float(ci_low),
    "diff_ci_high": float(ci_high),
    "chi2": float(chi2),
    "chi2_p": float(p_chi),
    "logit_unadj_coef_female": float(coef_unadj),
    "logit_unadj_or_female": float(or_unadj),
    "logit_unadj_or_ci_low": float(ci_unadj[0]),
    "logit_unadj_or_ci_high": float(ci_unadj[1]),
    "logit_unadj_p": float(p_unadj),
    "logit_adj_coef_female": float(coef_adj),
    "logit_adj_or_female": float(or_adj),
    "logit_adj_or_ci_low": float(ci_adj[0]),
    "logit_adj_or_ci_high": float(ci_adj[1]),
    "logit_adj_p": float(p_adj),
}

print(json.dumps(summary, indent=2))
