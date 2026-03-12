import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Variables
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['loc_adv'] = _df['dist_other'] - _df['dist_focal']  # positive means contest closer to focal than other

# Scale location to 100m for interpretability
_df['loc_adv_100'] = _df['loc_adv'] / 100.0

# Logistic regression
X = _df[['rel_size', 'loc_adv_100']]
X = sm.add_constant(X)
y = _df['win']

model = sm.Logit(y, X)
res = model.fit(disp=False)

# Output summary stats
print('N:', len(_df))
print('Win rate:', _df['win'].mean())
print(res.summary())

# Also check simple correlations (point-biserial via logistic? compute group means)
for col in ['rel_size', 'loc_adv_100']:
    print(col, 'mean if win=1:', _df.loc[_df['win']==1, col].mean(), 'mean if win=0:', _df.loc[_df['win']==0, col].mean())

# Predicted probabilities for 1-unit changes
params = res.params
conf = res.conf_int()
print('params:', params)
print('conf_int:', conf)

