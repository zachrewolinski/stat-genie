import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest

# Load data
_df = pd.read_csv('mortgage.csv')

# Ensure accept column
if 'accept' not in _df.columns and 'deny' in _df.columns:
    _df['accept'] = 1 - _df['deny']

# Relevant columns
cols = [
    'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value',
    'denied_PMI'
]

available_cols = [c for c in cols if c in _df.columns]

df = _df[available_cols].copy()

# Drop missing values in required columns
required_basic = ['accept', 'female']
for c in required_basic:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# Coerce to numeric
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df_basic = df.dropna(subset=['accept', 'female']).copy()

# Acceptance rates by gender
rates = df_basic.groupby('female')['accept'].agg(['mean', 'count'])

# Two-proportion z-test
count = (rates['mean'] * rates['count']).round().astype(int)
counts = count.values
nobs = rates['count'].values
stat, pval = proportions_ztest(counts, nobs)

# Unadjusted logistic regression with robust SEs
model_unadj = smf.glm('accept ~ female', data=df_basic, family=sm.families.Binomial()).fit(cov_type='HC1')

# Adjusted logistic regression
control_cols = [c for c in cols if c not in ['accept', 'female'] and c in df.columns]
model_adj = None
if control_cols:
    df_adj = df.dropna(subset=['accept', 'female'] + control_cols).copy()
    formula = 'accept ~ female + ' + ' + '.join(control_cols)
    model_adj = smf.glm(formula, data=df_adj, family=sm.families.Binomial()).fit(cov_type='HC1')
else:
    df_adj = df_basic.copy()

# Extract results
coef_unadj = model_unadj.params['female']
se_unadj = model_unadj.bse['female']
p_unadj = model_unadj.pvalues['female']
OR_unadj = float(np.exp(coef_unadj))

adj_results = None
if model_adj is not None:
    coef_adj = model_adj.params['female']
    se_adj = model_adj.bse['female']
    p_adj = model_adj.pvalues['female']
    OR_adj = float(np.exp(coef_adj))
    adj_results = (coef_adj, se_adj, p_adj, OR_adj, len(df_adj))

# Print summary
print('N total:', len(df_basic))
print('Acceptance rates by female (0=male,1=female):')
print(rates)
print('Two-proportion z-test p-value:', pval)
print('Unadjusted logit (female): coef=', coef_unadj, 'se=', se_unadj, 'p=', p_unadj, 'OR=', OR_unadj)
if adj_results:
    coef_adj, se_adj, p_adj, OR_adj, n_adj = adj_results
    print('Adjusted logit (female): coef=', coef_adj, 'se=', se_adj, 'p=', p_adj, 'OR=', OR_adj, 'N=', n_adj)
