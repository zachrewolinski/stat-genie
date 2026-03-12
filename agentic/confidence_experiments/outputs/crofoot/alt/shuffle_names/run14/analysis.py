import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Load data
path = 'crofoot.csv'
df = pd.read_csv(path)

# Inspect whether totals match male+female
# Columns: dist_focal (males focal?), other (females focal?) -> f_other (total focal?)
# Columns: focal (males other?), f_focal (females other?) -> win (total other?)

df['focal_size_from_components'] = df['dist_focal'] + df['other']
df['other_size_from_components'] = df['focal'] + df['f_focal']

# Compare to f_other and win
focal_match = (df['focal_size_from_components'] == df['f_other']).mean()
other_match = (df['other_size_from_components'] == df['win']).mean()

# Define outcome and predictors
# Outcome: m_focal (binary)
# Relative group size: focal total - other total
# Contest location: distance from center of home range for focal and other; perhaps relative distance
# Columns m_other and n_focal look like distances (meters); check which corresponds to focal/other
# Use difference (focal distance - other distance) and focal distance as location metrics

# m_other description: distance of focal from center
# n_focal description: distance of other from center

df['rel_group_size'] = df['f_other'] - df['win']
df['rel_distance'] = df['m_other'] - df['n_focal']

# Standardize predictors for comparability
for col in ['rel_group_size', 'rel_distance']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression
model = logit('m_focal ~ rel_group_size_z + rel_distance_z', data=df).fit(disp=False)

# Also include absolute distances for robustness
model_abs = logit('m_focal ~ rel_group_size_z + m_other + n_focal', data=df).fit(disp=False)

# Print summary stats
print('focal_size_match_rate', focal_match)
print('other_size_match_rate', other_match)
print(model.summary())
print(model_abs.summary())

# Simple correlation check
print('Correlation rel_group_size vs outcome', np.corrcoef(df['rel_group_size'], df['m_focal'])[0,1])
print('Correlation rel_distance vs outcome', np.corrcoef(df['rel_distance'], df['m_focal'])[0,1])

# Compute predicted effect sizes

# Get odds ratios and p-values
params = model.params
pvals = model.pvalues
or_vals = np.exp(params)
print('Odds ratios', or_vals)
print('p-values', pvals)
