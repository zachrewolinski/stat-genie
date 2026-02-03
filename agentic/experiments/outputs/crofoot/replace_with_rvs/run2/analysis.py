import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Feature engineering: relative group size and relative contest location
# rel_size > 0 means focal group is larger
# rel_dist > 0 means focal group is closer to its home-range center than the other group
_df['rel_size'] = _df['n_focal'] - _df['n_other']
_df['rel_dist'] = _df['dist_other'] - _df['dist_focal']

# Logistic regression: win ~ rel_size + rel_dist
X = sm.add_constant(_df[['rel_size', 'rel_dist']])
model = sm.Logit(_df['win'], X).fit(disp=False)

# Also compute a simple descriptive win rate by advantage signs
_df['larger'] = _df['rel_size'] > 0
_df['closer'] = _df['rel_dist'] > 0
win_rate_by_size = _df.groupby('larger')['win'].mean()
win_rate_by_loc = _df.groupby('closer')['win'].mean()

# Print results for record
print('Logit model: win ~ rel_size + rel_dist')
print(model.summary())
print('\nWin rate by larger (rel_size>0):')
print(win_rate_by_size)
print('\nWin rate by closer (rel_dist>0):')
print(win_rate_by_loc)

# Save a short table of coefficients and p-values
coef_table = pd.DataFrame({
    'coef': model.params,
    'p_value': model.pvalues,
    'odds_ratio': np.exp(model.params)
})
coef_table.to_csv('model_coefficients.csv')
