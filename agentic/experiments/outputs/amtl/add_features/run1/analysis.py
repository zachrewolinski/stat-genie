import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns and drop missing values
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = _df[cols].dropna().copy()

# Ensure valid binomial counts
if not (df['num_amtl'] <= df['sockets']).all():
    raise ValueError('num_amtl exceeds sockets for some rows')

# Human indicator vs non-human primates
# (Pan, Pongo, Papio are the non-human genera in this dataset)
df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Binomial GLM on AMTL proportion with trials as weights
# Equivalent to modeling counts with binomial denominator

df['prop'] = df['num_amtl'] / df['sockets']

model = smf.glm(
    'prop ~ human + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets'],
).fit()

# Extract human effect and its odds ratio with 95% CI
coef = model.params['human']
pval = model.pvalues['human']
ci_low, ci_high = model.conf_int().loc['human']
odds_ratio = float(np.exp(coef))
ci_or_low, ci_or_high = float(np.exp(ci_low)), float(np.exp(ci_high))

print(model.summary())
print('\nHuman coefficient:', coef)
print('Human p-value:', pval)
print('Human OR:', odds_ratio)
print('Human OR 95% CI:', (ci_or_low, ci_or_high))
