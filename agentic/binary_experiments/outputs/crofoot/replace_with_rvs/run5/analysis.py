import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Feature engineering
_df['rel_size'] = _df['n_focal'] - _df['n_other']
# Positive values mean the contest is closer to focal group's home range center
_df['location_adv'] = _df['dist_other'] - _df['dist_focal']

# Prepare design matrix
X = _df[['rel_size', 'location_adv']]
X = sm.add_constant(X)
y = _df['win']

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Summaries for reporting
params = result.params
pvalues = result.pvalues

print('Logit results:')
print(result.summary())

print('\nCoefficients:')
print(params)
print('\nP-values:')
print(pvalues)

# Also compute odds ratios for interpretability
odds_ratios = np.exp(params)
print('\nOdds ratios:')
print(odds_ratios)
