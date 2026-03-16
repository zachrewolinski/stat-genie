import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('crofoot.csv')

# Select relevant columns
cols = ['win','n_focal','n_other','dist_focal','dist_other']
# Drop rows with missing values in relevant columns
sub = df[cols].dropna()

# Create relative measures
sub['rel_size'] = sub['n_focal'] - sub['n_other']
sub['rel_location'] = sub['dist_other'] - sub['dist_focal']  # positive => contest closer to focal

# Standardize predictors for interpretability
sub['rel_size_z'] = (sub['rel_size'] - sub['rel_size'].mean())/sub['rel_size'].std(ddof=0)
sub['rel_location_z'] = (sub['rel_location'] - sub['rel_location'].mean())/sub['rel_location'].std(ddof=0)

# Logistic regression
model = smf.logit('win ~ rel_size_z + rel_location_z', data=sub).fit(disp=False)

# Also test each predictor alone
model_size = smf.logit('win ~ rel_size_z', data=sub).fit(disp=False)
model_loc = smf.logit('win ~ rel_location_z', data=sub).fit(disp=False)

print('N', len(sub))
print('Mean win', sub['win'].mean())
print('\nFull model')
print(model.summary2().tables[1])
print('\nPseudo R2', model.prsquared)
print('\nSize-only')
print(model_size.summary2().tables[1])
print('\nLoc-only')
print(model_loc.summary2().tables[1])

# compute predicted win probability difference for 1 SD increase
params = model.params
# baseline at mean predictors
base = params['Intercept']

def logistic(x):
    return 1/(1+np.exp(-x))

p_base = logistic(base)
# increase rel_size by 1 SD
p_size = logistic(base + params['rel_size_z'])
# increase rel_location by 1 SD
p_loc = logistic(base + params['rel_location_z'])
# increase both
p_both = logistic(base + params['rel_size_z'] + params['rel_location_z'])
print('\nPredicted prob at mean:', p_base)
print('Prob +1SD size:', p_size, 'delta', p_size - p_base)
print('Prob +1SD location:', p_loc, 'delta', p_loc - p_base)
print('Prob +1SD both:', p_both, 'delta', p_both - p_base)
