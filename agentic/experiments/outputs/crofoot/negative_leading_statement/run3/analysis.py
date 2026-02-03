import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Relative group size (focal minus other)
_df['rel_size'] = _df['n_focal'] - _df['n_other']

# Contest location: relative distance to home-range centers
# Positive means contest is closer to focal center than other center
_df['rel_location'] = _df['dist_other'] - _df['dist_focal']

# Standardize predictors for stable estimation
for col in ['rel_size', 'rel_location']:
    _df[col + '_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Logistic regression
X = _df[['rel_size_z', 'rel_location_z']]
X = sm.add_constant(X)
y = _df['win']
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Save results for inspection
summary = result.summary2().as_text()
with open('analysis_results.txt', 'w') as f:
    f.write(summary)

# Also compute simple win rate by sign of predictors
_df['rel_size_pos'] = _df['rel_size'] > 0
_df['rel_loc_pos'] = _df['rel_location'] > 0

win_rate_rel_size = _df.groupby('rel_size_pos')['win'].mean()
win_rate_rel_loc = _df.groupby('rel_loc_pos')['win'].mean()

with open('analysis_descriptives.txt', 'w') as f:
    f.write('Win rate by rel_size positive (focal larger):\n')
    f.write(win_rate_rel_size.to_string())
    f.write('\n\nWin rate by rel_location positive (closer to focal center):\n')
    f.write(win_rate_rel_loc.to_string())

print(summary)
print('\nWin rates by sign:')
print(win_rate_rel_size)
print(win_rate_rel_loc)
