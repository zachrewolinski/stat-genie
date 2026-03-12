import pandas as pd
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Map columns based on metadata inference
# genus -> AMTL measure (noisy count/logit)
# age -> sockets count
# pop -> age at death
# stdev_age -> prob male (sex)
# sockets -> tooth class
# tooth_class -> genus/species

_df['is_human'] = (_df['tooth_class'] == 'Homo sapiens').astype(int)
_df['amtl_rate'] = _df['genus'] / _df['age']

# Model 1: outcome = genus (AMTL measure)
formula1 = 'genus ~ is_human + pop + stdev_age + C(sockets)'
model1 = smf.ols(formula1, data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['prob_male']})

# Model 2: outcome = amtl_rate (rate)
formula2 = 'amtl_rate ~ is_human + pop + stdev_age + C(sockets)'
model2 = smf.ols(formula2, data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['prob_male']})

# Simple adjusted difference: predicted mean for human vs non-human at mean covariates and pooled tooth_class distribution
# We'll compute marginal effect by setting is_human to 0/1 and averaging predictions.

def marginal_effect(model, data):
    d0 = data.copy()
    d1 = data.copy()
    d0['is_human'] = 0
    d1['is_human'] = 1
    pred0 = model.predict(d0).mean()
    pred1 = model.predict(d1).mean()
    return pred1 - pred0, pred1, pred0

me1 = marginal_effect(model1, _df)
me2 = marginal_effect(model2, _df)

print('Model1: genus outcome')
print(model1.summary().tables[1])
print('is_human coef', model1.params['is_human'], 'p', model1.pvalues['is_human'])
print('marginal diff human - nonhuman', me1)

print('\nModel2: amtl_rate outcome')
print(model2.summary().tables[1])
print('is_human coef', model2.params['is_human'], 'p', model2.pvalues['is_human'])
print('marginal diff human - nonhuman', me2)
