import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Feature engineering
_df['size_diff'] = _df['n_focal'] - _df['n_other']
# Positive when focal is closer to its home range center than the other group
_df['dist_diff'] = _df['dist_other'] - _df['dist_focal']

# Basic descriptive summaries
win_rate = _df['win'].mean()
win_by_size_adv = _df.groupby(_df['size_diff'] > 0)['win'].mean()
win_by_home_adv = _df.groupby(_df['dist_diff'] > 0)['win'].mean()

# Logistic regression
X = _df[['size_diff', 'dist_diff']]
X = sm.add_constant(X)
model = sm.Logit(_df['win'], X)
result = model.fit(disp=False)

# Also standardized coefficients for comparability
X_std = _df[['size_diff', 'dist_diff']].astype(float)
X_std = (X_std - X_std.mean()) / X_std.std(ddof=0)
X_std = sm.add_constant(X_std)
model_std = sm.Logit(_df['win'], X_std)
result_std = model_std.fit(disp=False)

print('Win rate:', win_rate)
print('Win rate by size advantage (size_diff>0):')
print(win_by_size_adv)
print('Win rate by home-range advantage (dist_diff>0):')
print(win_by_home_adv)
print('\nLogit coefficients (unstandardized):')
print(result.summary2().tables[1])
print('\nLogit coefficients (standardized):')
print(result_std.summary2().tables[1])
