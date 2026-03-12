import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('crofoot.csv')

# Variables for research question
# Relative group size (focal - other) and contest location (relative distance from each group's home range center)
# Positive dist_diff means focal is farther from its own center than other group is from its center,
# so contest is closer to the other group's center (location disadvantage for focal).

df['size_diff'] = df['n_focal'] - df['n_other']
df['dist_diff'] = df['dist_focal'] - df['dist_other']

# Standardize predictors for comparability
for col in ['size_diff', 'dist_diff']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Fit logistic regression
model = smf.glm('win ~ size_diff_z + dist_diff_z', data=df, family=sm.families.Binomial()).fit()

# Extract results
params = model.params
pvalues = model.pvalues

# Odds ratios for 1 SD increase
odds_ratios = np.exp(params)

# Descriptive win rates by relative advantage
size_adv = df['size_diff'] > 0
loc_adv = df['dist_diff'] < 0  # focal closer to its center

win_rate_size_adv = df.loc[size_adv, 'win'].mean()
win_rate_size_disadv = df.loc[~size_adv, 'win'].mean()

win_rate_loc_adv = df.loc[loc_adv, 'win'].mean()
win_rate_loc_disadv = df.loc[~loc_adv, 'win'].mean()

summary = {
    'n': int(df.shape[0]),
    'coef_size_diff_z': float(params['size_diff_z']),
    'p_size_diff_z': float(pvalues['size_diff_z']),
    'odds_ratio_size_diff_z': float(odds_ratios['size_diff_z']),
    'coef_dist_diff_z': float(params['dist_diff_z']),
    'p_dist_diff_z': float(pvalues['dist_diff_z']),
    'odds_ratio_dist_diff_z': float(odds_ratios['dist_diff_z']),
    'win_rate_size_adv': float(win_rate_size_adv),
    'win_rate_size_disadv': float(win_rate_size_disadv),
    'win_rate_loc_adv': float(win_rate_loc_adv),
    'win_rate_loc_disadv': float(win_rate_loc_disadv),
}

print(json.dumps(summary, indent=2))
