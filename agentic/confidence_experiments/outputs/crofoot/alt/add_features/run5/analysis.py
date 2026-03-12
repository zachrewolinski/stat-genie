import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('crofoot.csv')

# Focus on relevant columns
needed = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
missing = [c for c in needed if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

# Drop rows with missing values in key columns
sub = df[needed].dropna().copy()

# Feature engineering
sub['size_diff'] = sub['n_focal'] - sub['n_other']
sub['size_ratio'] = sub['n_focal'] / sub['n_other']
sub['loc_adv'] = sub['dist_other'] - sub['dist_focal']  # positive means closer to focal range center
sub['focal_closer'] = (sub['dist_focal'] < sub['dist_other']).astype(int)

# Logistic regression: win ~ size_diff + loc_adv
model = smf.logit('win ~ size_diff + loc_adv', data=sub).fit(disp=0)

# Alternative with size_ratio
model_ratio = smf.logit('win ~ size_ratio + loc_adv', data=sub).fit(disp=0)

# Descriptive win rates
win_rate = sub['win'].mean()
win_rate_bigger = sub.loc[sub['size_diff'] > 0, 'win'].mean()
win_rate_smaller = sub.loc[sub['size_diff'] < 0, 'win'].mean()
win_rate_equal = sub.loc[sub['size_diff'] == 0, 'win'].mean()

win_rate_focal_closer = sub.loc[sub['focal_closer'] == 1, 'win'].mean()
win_rate_focal_farther = sub.loc[sub['focal_closer'] == 0, 'win'].mean()

# Print summary
print('N:', len(sub))
print('Overall win rate:', win_rate)
print('Win rate focal bigger:', win_rate_bigger)
print('Win rate focal smaller:', win_rate_smaller)
print('Win rate equal size:', win_rate_equal)
print('Win rate focal closer:', win_rate_focal_closer)
print('Win rate focal farther:', win_rate_focal_farther)

print('\nLogit win ~ size_diff + loc_adv')
print(model.summary2().tables[1])

print('\nLogit win ~ size_ratio + loc_adv')
print(model_ratio.summary2().tables[1])

# Save key results for downstream use
out = {
    'n': int(len(sub)),
    'win_rate': float(win_rate),
    'win_rate_bigger': float(win_rate_bigger) if not np.isnan(win_rate_bigger) else None,
    'win_rate_smaller': float(win_rate_smaller) if not np.isnan(win_rate_smaller) else None,
    'win_rate_equal': float(win_rate_equal) if not np.isnan(win_rate_equal) else None,
    'win_rate_focal_closer': float(win_rate_focal_closer) if not np.isnan(win_rate_focal_closer) else None,
    'win_rate_focal_farther': float(win_rate_focal_farther) if not np.isnan(win_rate_focal_farther) else None,
    'coef_size_diff': float(model.params['size_diff']),
    'p_size_diff': float(model.pvalues['size_diff']),
    'coef_loc_adv': float(model.params['loc_adv']),
    'p_loc_adv': float(model.pvalues['loc_adv']),
    'coef_size_ratio': float(model_ratio.params['size_ratio']),
    'p_size_ratio': float(model_ratio.pvalues['size_ratio']),
    'coef_loc_adv_ratio': float(model_ratio.params['loc_adv']),
    'p_loc_adv_ratio': float(model_ratio.pvalues['loc_adv']),
}

pd.DataFrame([out]).to_csv('analysis_summary.csv', index=False)
