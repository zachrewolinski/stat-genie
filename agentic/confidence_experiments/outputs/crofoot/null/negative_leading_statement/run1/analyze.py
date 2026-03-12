import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Basic checks
print('rows', len(df))
print(df.head())

# Define predictors
# Relative group size (focal - other)
df['rel_size'] = df['n_focal'] - df['n_other']

# Contest location: difference in distance to home range center (other - focal)
# If focal is closer to its own center than other, focal likely on home turf.
# We'll use dist_other - dist_focal so positive means focal closer than other.
df['rel_location'] = df['dist_other'] - df['dist_focal']

# Another possible location metric: focal distance to its center (smaller is better)
# We'll keep for sensitivity.

# Fit logistic regression: win ~ rel_size + rel_location
X = sm.add_constant(df[['rel_size', 'rel_location']])
model = sm.Logit(df['win'], X)
res = model.fit(disp=False)
print(res.summary())

# Fit logistic regression with just rel_size
X1 = sm.add_constant(df[['rel_size']])
res1 = sm.Logit(df['win'], X1).fit(disp=False)
print(res1.summary())

# Fit logistic regression with just rel_location
X2 = sm.add_constant(df[['rel_location']])
res2 = sm.Logit(df['win'], X2).fit(disp=False)
print(res2.summary())

# Sensitivity: use dist_focal and dist_other separately
X3 = sm.add_constant(df[['rel_size', 'dist_focal', 'dist_other']])
res3 = sm.Logit(df['win'], X3).fit(disp=False)
print(res3.summary())

# Another location metric: focal distance to its center alone
X4 = sm.add_constant(df[['rel_size', 'dist_focal']])
res4 = sm.Logit(df['win'], X4).fit(disp=False)
print(res4.summary())

# Descriptive win rates by rel_size sign and by rel_location sign
print('\nWin rate by rel_size sign:')
print(df.groupby(np.sign(df['rel_size']))['win'].mean())
print('Counts by rel_size sign:')
print(df.groupby(np.sign(df['rel_size']))['win'].count())

print('\nWin rate by rel_location sign (positive means focal closer):')
print(df.groupby(np.sign(df['rel_location']))['win'].mean())
print('Counts by rel_location sign:')
print(df.groupby(np.sign(df['rel_location']))['win'].count())

# Correlation checks
print('\nCorrelation rel_size vs win:', df['rel_size'].corr(df['win']))
print('Correlation rel_location vs win:', df['rel_location'].corr(df['win']))

# Quick logistic regression with interaction (rel_size * rel_location)
df['interaction'] = df['rel_size'] * df['rel_location']
X5 = sm.add_constant(df[['rel_size', 'rel_location', 'interaction']])
res5 = sm.Logit(df['win'], X5).fit(disp=False)
print(res5.summary())

# Print odds ratios for main model
or_ci = np.exp(res.conf_int())
ors = np.exp(res.params)
print('\nOdds ratios (main model):')
print(pd.DataFrame({'OR': ors, 'CI_low': or_ci[0], 'CI_high': or_ci[1]}))
