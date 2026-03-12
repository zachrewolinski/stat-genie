import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
DF_PATH = 'mortgage.csv'
df = pd.read_csv(DF_PATH)

# Identify variables based on data inspection and consistency checks
# "deny" is approval (1=approved) and "self_employed" is denial (1=denied), perfect complements
approval = df['deny']

# "denied_PMI" has plausible female share (~0.21) and matches expected binary gender distribution
female = df['denied_PMI']

# Basic checks (allow missing in female)
assert set(female.dropna().unique()) <= {0, 1}
assert set(approval.unique()) <= {0, 1}

# Drop rows with missing female or key covariates for consistent analysis
exclude_cols = {'deny', 'self_employed', 'bad_history'}
X_cols = [c for c in df.columns if c not in exclude_cols]
needed_cols = ['deny', 'denied_PMI'] + X_cols
# Deduplicate while preserving order
needed_cols = list(dict.fromkeys(needed_cols))
df_clean = df[needed_cols].dropna()

approval_clean = df_clean['deny']
female_clean = df_clean['denied_PMI']

# Approval rates by gender
rate_female = approval_clean[female_clean == 1].mean()
rate_male = approval_clean[female_clean == 0].mean()
rate_diff = rate_female - rate_male

# Contingency table for chi-square test
ct = pd.crosstab(female_clean, approval_clean)
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Logistic regression: approval ~ female + controls
X = df_clean[X_cols].copy()
X = sm.add_constant(X, has_constant='add')

# Fit model (use GLM binomial for stability)
model = sm.GLM(approval_clean, X, family=sm.families.Binomial())
res = model.fit()

# Extract female coefficient
coef_female = res.params['denied_PMI']
se_female = res.bse['denied_PMI']
# Wald test
z_female = coef_female / se_female
p_female = 2 * (1 - stats.norm.cdf(abs(z_female)))

# Marginal effect approximation at mean: derivative of logit = p*(1-p)*beta
p_mean = approval_clean.mean()
me_female = p_mean * (1 - p_mean) * coef_female

# Unadjusted logit: approval ~ female only
X_unadj = sm.add_constant(df_clean[['denied_PMI']], has_constant='add')
model_unadj = sm.GLM(approval_clean, X_unadj, family=sm.families.Binomial())
res_unadj = model_unadj.fit()
coef_female_unadj = res_unadj.params['denied_PMI']
se_female_unadj = res_unadj.bse['denied_PMI']
z_female_unadj = coef_female_unadj / se_female_unadj
p_female_unadj = 2 * (1 - stats.norm.cdf(abs(z_female_unadj)))

results = {
    'n': len(df_clean),
    'approval_rate_overall': approval_clean.mean(),
    'approval_rate_female': rate_female,
    'approval_rate_male': rate_male,
    'approval_rate_diff_female_minus_male': rate_diff,
    'chi2_p_value': p_chi,
    'logit_unadjusted_coef_female': coef_female_unadj,
    'logit_unadjusted_p_value_female': p_female_unadj,
    'logit_coef_female': coef_female,
    'logit_se_female': se_female,
    'logit_p_value_female': p_female,
    'logit_marginal_effect_at_mean': me_female,
}

print(results)
