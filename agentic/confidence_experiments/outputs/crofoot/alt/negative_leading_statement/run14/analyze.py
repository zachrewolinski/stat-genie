import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Derived variables
# Relative group size (focal - other)
df['rel_size'] = df['n_focal'] - df['n_other']
# Relative location (other distance - focal distance): positive means contest closer to focal home range center
# because other is farther from its center than focal is from its own center.
df['rel_location'] = df['dist_other'] - df['dist_focal']

# Standardize predictors for interpretability
for col in ['rel_size', 'rel_location', 'dist_focal', 'dist_other', 'n_focal', 'n_other']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression: win ~ rel_size + rel_location
X = df[['rel_size_z', 'rel_location_z']]
X = sm.add_constant(X)
model = sm.Logit(df['win'], X).fit(disp=False)

# Alternative model: win ~ dist_focal + dist_other + rel_size
X2 = df[['dist_focal_z', 'dist_other_z', 'rel_size_z']]
X2 = sm.add_constant(X2)
model2 = sm.Logit(df['win'], X2).fit(disp=False)

# Simple group size effect test
# Use t-test comparing rel_size between wins and losses
rel_size_win = df.loc[df['win'] == 1, 'rel_size']
rel_size_loss = df.loc[df['win'] == 0, 'rel_size']
size_t = stats.ttest_ind(rel_size_win, rel_size_loss, equal_var=False)

# Simple location effect test
rel_loc_win = df.loc[df['win'] == 1, 'rel_location']
rel_loc_loss = df.loc[df['win'] == 0, 'rel_location']
loc_t = stats.ttest_ind(rel_loc_win, rel_loc_loss, equal_var=False)

# Summaries
print('n:', len(df))
print('win rate:', df['win'].mean())
print('\nLogit: win ~ rel_size_z + rel_location_z')
print(model.summary())
print('\nLogit: win ~ dist_focal_z + dist_other_z + rel_size_z')
print(model2.summary())
print('\nT-test rel_size win vs loss:', size_t)
print('T-test rel_location win vs loss:', loc_t)

# Compute marginal effects for primary model
margeff = model.get_margeff(at='mean')
print('\nMarginal effects (at mean):')
print(margeff.summary())

# Provide odds ratios
params = model.params
conf = model.conf_int()
odds = np.exp(params)
conf_odds = np.exp(conf)
print('\nOdds ratios (primary model):')
print(pd.DataFrame({'odds_ratio': odds, 'ci_lower': conf_odds[0], 'ci_upper': conf_odds[1]}))
