import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Map variables
_df['is_human'] = (_df['tooth_class'] == 'Homo sapiens').astype(int)

# Build model: response ~ human + age at death + sex + tooth class + sockets count
# age at death: pop
# sex estimate: stdev_age (0-1)
# sockets count: age
# tooth class: sockets

model = smf.ols(
    'genus ~ is_human + pop + stdev_age + age + C(sockets)',
    data=_df
).fit(cov_type='HC3')

print(model.summary())

# Extract human effect
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

print('\nHuman effect coefficient:', coef)
print('Human effect SE:', se)
print('Human effect p-value:', pval)

# Also compare humans to each non-human genus via full categorical model
model2 = smf.ols(
    'genus ~ C(tooth_class) + pop + stdev_age + age + C(sockets)',
    data=_df
).fit(cov_type='HC3')
print('\nCategorical genus model summary:')
print(model2.summary())

# Extract coefficients for Homo sapiens vs baseline (alphabetical baseline)
params = model2.params
pvals = model2.pvalues

for name in params.index:
    if 'C(tooth_class)' in name:
        print(name, params[name], pvals[name])
