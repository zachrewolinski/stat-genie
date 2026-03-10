import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
_df = pd.read_csv('crofoot.csv')

# Compute relative measures
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_location'] = _df['dist_other'] - _df['dist_focal']  # positive => focal closer to its center

# Basic summaries
summary = {
    'n': len(_df),
    'win_rate': _df['win'].mean(),
    'rel_size_mean': _df['rel_size'].mean(),
    'rel_location_mean': _df['rel_location'].mean(),
    'rel_size_sd': _df['rel_size'].std(),
    'rel_location_sd': _df['rel_location'].std(),
}

# Logistic regression
X = _df[['rel_size', 'rel_location']]
X = sm.add_constant(X)
model = sm.Logit(_df['win'], X).fit(disp=False)

# Also scale rel_location per 100 meters for interpretability
_df['rel_location_100'] = _df['rel_location'] / 100.0
X2 = sm.add_constant(_df[['rel_size', 'rel_location_100']])
model2 = sm.Logit(_df['win'], X2).fit(disp=False)

# Print results
print('SUMMARY', summary)
print('\nLOGIT rel_size + rel_location')
print(model.summary())
print('\nLOGIT rel_size + rel_location_100 (per 100m)')
print(model2.summary())

# Odds ratios
params = model2.params
conf = model2.conf_int()
ors = np.exp(params)
conf_or = np.exp(conf)
print('\nODDS RATIOS (per 100m for rel_location_100)')
for k in params.index:
    print(k, ors[k], conf_or.loc[k,0], conf_or.loc[k,1])
