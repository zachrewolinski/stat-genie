import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data


df = pd.read_csv('crofoot.csv')

# Rename for clarity
cols = df.columns.tolist()

# outcome
outcome = 'feature4'

# compute relative group size (focal - other)
df['rel_size'] = df['feature7'] - df['feature8']

# compute relative location: focal distance from its home range center minus other distance
# Negative -> focal closer to its home range center than other is to its own (relative home advantage)
df['rel_location'] = df['feature5'] - df['feature6']

# Also compute ratio versions for robustness
# Avoid division by zero (none should be 0)
df['size_ratio'] = df['feature7'] / df['feature8']
df['location_ratio'] = df['feature5'] / df['feature6']

# Logistic regression with rel_size and rel_location
X = sm.add_constant(df[['rel_size', 'rel_location']])
model = sm.Logit(df[outcome], X).fit(disp=False)

# Logistic regression with ratios
Xr = sm.add_constant(df[['size_ratio', 'location_ratio']])
model_ratio = sm.Logit(df[outcome], Xr).fit(disp=False)

# Simple descriptive statistics
# Win rate when focal larger vs smaller
win_rate_larger = df.loc[df['rel_size'] > 0, outcome].mean()
win_rate_equal = df.loc[df['rel_size'] == 0, outcome].mean()
win_rate_smaller = df.loc[df['rel_size'] < 0, outcome].mean()

# Win rate when focal is closer to its home center than other (rel_location < 0)
win_rate_home_adv = df.loc[df['rel_location'] < 0, outcome].mean()
win_rate_home_disadv = df.loc[df['rel_location'] > 0, outcome].mean()
win_rate_home_equal = df.loc[df['rel_location'] == 0, outcome].mean()

# Print summaries
print('N:', len(df))
print('\nLogit model (rel_size, rel_location):')
print(model.summary())

print('\nLogit model (size_ratio, location_ratio):')
print(model_ratio.summary())

print('\nWin rates by relative size:')
print('larger:', win_rate_larger, 'equal:', win_rate_equal, 'smaller:', win_rate_smaller)

print('\nWin rates by relative location:')
print('home_adv (rel_location < 0):', win_rate_home_adv)
print('home_disadv (rel_location > 0):', win_rate_home_disadv)
print('home_equal (rel_location == 0):', win_rate_home_equal)

# also compute odds ratios and confidence intervals for rel_size and rel_location
params = model.params
conf = model.conf_int()
conf.columns = ['2.5%', '97.5%']

or_params = np.exp(params)
or_conf = np.exp(conf)

print('\nOdds ratios (rel_size, rel_location):')
print(pd.DataFrame({'OR': or_params, 'OR_2.5%': or_conf['2.5%'], 'OR_97.5%': or_conf['97.5%']}))

# for ratio model
params_r = model_ratio.params
conf_r = model_ratio.conf_int()
conf_r.columns = ['2.5%', '97.5%']

or_params_r = np.exp(params_r)
or_conf_r = np.exp(conf_r)

print('\nOdds ratios (ratio model):')
print(pd.DataFrame({'OR': or_params_r, 'OR_2.5%': or_conf_r['2.5%'], 'OR_97.5%': or_conf_r['97.5%']}))
