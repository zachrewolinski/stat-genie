import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Define variables
_df = _df.copy()
_df['is_human'] = (_df['tooth_class'] == 'Homo sapiens').astype(int)

# OLS with cluster-robust SE by specimen id (prob_male)
model = smf.ols('genus ~ is_human + pop + stdev_age + C(sockets)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['prob_male']}
)

print(model.summary())

# Extract coefficient and p-value for is_human
coef = model.params['is_human']
pval = model.pvalues['is_human']
print('is_human coef', coef, 'p', pval)

# compute adjusted mean difference (same as coef) and 95% CI
ci = model.conf_int().loc['is_human']
print('is_human 95% CI', ci.values)

# Also fit model with full genus categories for comparison
model2 = smf.ols('genus ~ C(tooth_class) + pop + stdev_age + C(sockets)', data=_df).fit(
    cov_type='cluster', cov_kwds={'groups': _df['prob_male']}
)
print(model2.summary())

