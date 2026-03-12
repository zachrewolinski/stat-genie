import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Rename columns for clarity
col_map = {
    'feature4': 'focal_win',
    'feature5': 'focal_dist_center',
    'feature6': 'other_dist_center',
    'feature7': 'focal_group_size',
    'feature8': 'other_group_size',
}
df = df.rename(columns=col_map)

# Derived predictors
# Relative group size: focal - other
# Relative location: other distance - focal distance (positive means contest closer to focal center)
df['rel_group_size'] = df['focal_group_size'] - df['other_group_size']
df['rel_location'] = df['other_dist_center'] - df['focal_dist_center']

# Drop any missing values
model_df = df[['focal_win', 'rel_group_size', 'rel_location']].dropna()

# Standardize predictors for stable estimation and comparable effect sizes
X = model_df[['rel_group_size', 'rel_location']]
X_std = (X - X.mean()) / X.std(ddof=0)
X_std = sm.add_constant(X_std)

y = model_df['focal_win']

# Logistic regression
logit_model = sm.Logit(y, X_std)
result = logit_model.fit(disp=False)

# Also fit unstandardized for odds ratios per unit
X_un = sm.add_constant(X)
result_un = sm.Logit(y, X_un).fit(disp=False)

# Summaries
print('N:', len(model_df))
print('\nStandardized predictors (z) logistic regression:')
print(result.summary())

print('\nUnstandardized predictors logistic regression:')
print(result_un.summary())

# Compute odds ratios and CI for unstandardized model
params = result_un.params
conf = result_un.conf_int()
conf.columns = ['2.5%', '97.5%']

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

print('\nOdds ratios (per unit increase):')
for term in ['rel_group_size', 'rel_location']:
    or_val = odds_ratios[term]
    lo, hi = conf_or.loc[term]
    print(f'{term}: OR={or_val:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]')

# Simple correlation check
print('\nPoint-biserial correlations:')
for term in ['rel_group_size', 'rel_location']:
    corr = np.corrcoef(model_df[term], y)[0, 1]
    print(f'{term}: r={corr:.3f}')
