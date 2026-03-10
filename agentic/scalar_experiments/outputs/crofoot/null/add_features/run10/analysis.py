import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('crofoot.csv')

# Drop rows with missing in relevant cols
cols = ['win','n_focal','n_other','dist_focal','dist_other']
# ensure numeric
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

sub = df[cols].dropna().copy()

# Derived variables
sub['rel_size'] = sub['n_focal'] - sub['n_other']
sub['rel_dist'] = sub['dist_other'] - sub['dist_focal']  # positive => focal closer to its home-range center than other

# Standardize for model stability
sub['rel_size_z'] = (sub['rel_size'] - sub['rel_size'].mean())/sub['rel_size'].std(ddof=0)
sub['rel_dist_z'] = (sub['rel_dist'] - sub['rel_dist'].mean())/sub['rel_dist'].std(ddof=0)

# Logistic regression (GLM binomial)
model = smf.glm('win ~ rel_size_z + rel_dist_z', data=sub, family=sm.families.Binomial())
res = model.fit()

# Alternative model with raw variables
model_raw = smf.glm('win ~ rel_size + rel_dist', data=sub, family=sm.families.Binomial())
res_raw = model_raw.fit()

# Simple comparisons
sub['size_adv'] = (sub['rel_size'] > 0).astype(int)
sub['loc_adv'] = (sub['rel_dist'] > 0).astype(int)

size_adv_rate = sub.groupby('size_adv')['win'].mean()
loc_adv_rate = sub.groupby('loc_adv')['win'].mean()

# Output key results
print('N', len(sub))
print('win_rate', sub['win'].mean())
print('size_adv_rate', size_adv_rate.to_dict())
print('loc_adv_rate', loc_adv_rate.to_dict())
print('\nGLM standardized coefficients')
print(res.summary())
print('\nGLM raw coefficients')
print(res_raw.summary())

# also compute odds ratios for z model
params = res.params
conf = res.conf_int()
ors = np.exp(params)
conf_or = np.exp(conf)
print('\nOdds ratios (z model):')
print(pd.DataFrame({'OR': ors, 'CI_low': conf_or[0], 'CI_high': conf_or[1], 'p': res.pvalues}))
