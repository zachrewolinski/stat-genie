import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Clean: drop unnamed column if present
if 'Unnamed: 0' in _df.columns:
    df = _df.drop(columns=['Unnamed: 0'])
else:
    df = _df.copy()

n_total = len(df)

# Acceptance rate by gender
accept_rate_by_gender = df.groupby('female')['accept'].mean()
counts_by_gender = df.groupby('female')['accept'].agg(['count','sum'])

# Chi-square test of independence between female and accept
contingency = pd.crosstab(df['female'], df['accept'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

# Difference in proportions (female - male)
# proportion accepted
p_female = accept_rate_by_gender.loc[1.0] if 1.0 in accept_rate_by_gender.index else accept_rate_by_gender.loc[1]
p_male = accept_rate_by_gender.loc[0.0] if 0.0 in accept_rate_by_gender.index else accept_rate_by_gender.loc[0]
prop_diff = p_female - p_male

# Standard error for difference in proportions
n_female = counts_by_gender.loc[1, 'count']
accepted_female = counts_by_gender.loc[1, 'sum']

n_male = counts_by_gender.loc[0, 'count']
accepted_male = counts_by_gender.loc[0, 'sum']

se_diff = np.sqrt(p_female*(1-p_female)/n_female + p_male*(1-p_male)/n_male)
z = prop_diff / se_diff if se_diff > 0 else np.nan
p_z = 2*(1-stats.norm.cdf(abs(z))) if se_diff > 0 else np.nan

# Logistic regression: accept ~ female (unadjusted)
# Drop rows with missing in accept or female
unadj_df = df[['accept','female']].dropna()
X1 = sm.add_constant(unadj_df[['female']])
model1 = sm.GLM(unadj_df['accept'], X1, family=sm.families.Binomial()).fit()

# Adjusted model with controls
control_cols = ['female','black','housing_expense_ratio','self_employed','married',
                'mortgage_credit','consumer_credit','bad_history','PI_ratio',
                'loan_to_value','denied_PMI']

# drop rows with missing in controls
adj_df = df[control_cols + ['accept']].dropna()
X2 = sm.add_constant(adj_df[control_cols])
model2 = sm.GLM(adj_df['accept'], X2, family=sm.families.Binomial()).fit()

# Extract female coefficient, odds ratio, p-value
coef1 = model1.params['female']
or1 = np.exp(coef1)
pval1 = model1.pvalues['female']

coef2 = model2.params['female']
or2 = np.exp(coef2)
pval2 = model2.pvalues['female']

results = {
    'n_total': int(n_total),
    'n_unadj': int(len(unadj_df)),
    'n_adj': int(len(adj_df)),
    'accept_rate_by_gender': accept_rate_by_gender.to_dict(),
    'counts_by_gender': counts_by_gender.to_dict(),
    'contingency': contingency.to_dict(),
    'chi2': float(chi2),
    'p_chi': float(p_chi),
    'prop_diff': float(prop_diff),
    'z': float(z),
    'p_z': float(p_z),
    'logit_unadj_coef': float(coef1),
    'logit_unadj_or': float(or1),
    'logit_unadj_p': float(pval1),
    'logit_adj_coef': float(coef2),
    'logit_adj_or': float(or2),
    'logit_adj_p': float(pval2)
}

import json
print(json.dumps(results, indent=2))
