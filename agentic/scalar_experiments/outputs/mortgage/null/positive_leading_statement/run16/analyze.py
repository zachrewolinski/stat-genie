import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "mortgage.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning: drop rows with missing relevant columns
outcome = "accept"
key_var = "female"

# Identify relevant covariates based on data description
covariates = [
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

cols = [outcome, key_var] + covariates

df_model = df[cols].dropna()

# Ensure binary outcomes are 0/1

# Unadjusted acceptance rates by gender
rates = df_model.groupby(key_var)[outcome].mean()
counts = df_model.groupby(key_var)[outcome].agg(['count', 'sum'])

# Two-proportion z-test for difference in acceptance rates
# female=1 vs female=0
count1 = int(counts.loc[1, 'sum']) if 1 in counts.index else 0
nobs1 = int(counts.loc[1, 'count']) if 1 in counts.index else 0
count0 = int(counts.loc[0, 'sum']) if 0 in counts.index else 0
nobs0 = int(counts.loc[0, 'count']) if 0 in counts.index else 0

# two-sided z-test
if nobs0 > 0 and nobs1 > 0:
    stat, pval = sm.stats.proportions_ztest([count1, count0], [nobs1, nobs0])
else:
    stat, pval = np.nan, np.nan

# Logistic regression: accept ~ female (unadjusted)
formula_unadj = f"{outcome} ~ {key_var}"
model_unadj = smf.logit(formula_unadj, data=df_model).fit(disp=False)

# Logistic regression: accept ~ female + covariates (adjusted)
formula_adj = f"{outcome} ~ {key_var} + " + " + ".join(covariates)
model_adj = smf.logit(formula_adj, data=df_model).fit(disp=False)

# Extract effect of female
coef_unadj = model_unadj.params.get(key_var)
p_unadj = model_unadj.pvalues.get(key_var)

coef_adj = model_adj.params.get(key_var)
p_adj = model_adj.pvalues.get(key_var)

# Convert to odds ratios
or_unadj = float(np.exp(coef_unadj))
or_adj = float(np.exp(coef_adj))

# 95% CI for adjusted odds ratio
ci_adj = model_adj.conf_int().loc[key_var].to_numpy()
ci_adj_or = np.exp(ci_adj)

# Also compute marginal effect difference at means (optional)
try:
    marg_eff = model_adj.get_margeff(at='mean').summary_frame().loc[key_var]
    meff = float(marg_eff['dy/dx'])
    meff_p = float(marg_eff['P>|z|'])
except Exception:
    meff = np.nan
    meff_p = np.nan

results = {
    "n": int(df_model.shape[0]),
    "accept_rate_female0": float(rates.get(0, np.nan)),
    "accept_rate_female1": float(rates.get(1, np.nan)),
    "prop_z_p": float(pval) if pval == pval else None,
    "logit_unadj_coef": float(coef_unadj),
    "logit_unadj_or": or_unadj,
    "logit_unadj_p": float(p_unadj),
    "logit_adj_coef": float(coef_adj),
    "logit_adj_or": or_adj,
    "logit_adj_p": float(p_adj),
    "logit_adj_or_ci_low": float(ci_adj_or[0]),
    "logit_adj_or_ci_high": float(ci_adj_or[1]),
    "marginal_effect_mean": meff,
    "marginal_effect_p": meff_p,
}

print(json.dumps(results, indent=2))
