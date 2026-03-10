import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv('mortgage.csv')

# Basic counts
n_total = len(df)
counts = df['female'].value_counts().sort_index()

# Acceptance rates by gender (using available data)
rates = df.groupby('female')['accept'].mean()

# Two-proportion z-test (acceptance)
count_success = df.groupby('female')['accept'].sum()
count_total = df.groupby('female')['accept'].count()
stat, pval = proportions_ztest(count_success.values, count_total.values)

# Unadjusted logit: accept ~ female
cols_unadj = ['accept', 'female']
df_unadj = df[cols_unadj].dropna()
X_unadj = sm.add_constant(df_unadj[['female']])
model_unadj = sm.Logit(df_unadj['accept'], X_unadj).fit(disp=False)

# Adjusted logit with available covariates (excluding accept/deny and index)
covariates = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
cols_adj = ['accept'] + covariates
df_adj = df[cols_adj].dropna()
X_adj = sm.add_constant(df_adj[covariates])
model_adj = sm.Logit(df_adj['accept'], X_adj).fit(disp=False)

# Extract female coefficient stats
coef_unadj = model_unadj.params['female']
OR_unadj = np.exp(coef_unadj)

coef_adj = model_adj.params['female']
OR_adj = np.exp(coef_adj)

# 95% CI for odds ratios
ci_unadj = np.exp(model_unadj.conf_int().loc['female'].values)
ci_adj = np.exp(model_adj.conf_int().loc['female'].values)

print('N total:', n_total)
print('Counts female=0,1:', counts.to_dict())
print('Acceptance rates by female:', rates.to_dict())
print('Difference in acceptance rates (female - male):', rates.loc[1] - rates.loc[0])
print('Two-proportion z-test p-value:', pval)
print('Unadjusted logit female coef:', coef_unadj, 'OR:', OR_unadj, 'p:', model_unadj.pvalues['female'])
print('Unadjusted OR 95% CI:', ci_unadj)
print('Adjusted logit female coef:', coef_adj, 'OR:', OR_adj, 'p:', model_adj.pvalues['female'])
print('Adjusted OR 95% CI:', ci_adj)
print('Unadjusted N used:', len(df_unadj))
print('Adjusted N used:', len(df_adj))
