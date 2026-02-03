import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data

df = pd.read_csv('soccer.csv')

# Build skin tone composite (mean of two raters)

df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Define dark vs light by median split
median_skin = df['skin_tone'].median()

df['dark'] = (df['skin_tone'] > median_skin).astype(int)

# Outcome measures

df['any_red'] = (df['redCards'] > 0).astype(int)

# Descriptive stats

mean_red_by_group = df.groupby('dark')['redCards'].mean()
rate_any_red_by_group = df.groupby('dark')['any_red'].mean()

# Controls

control_cols = [
    'games', 'yellowCards', 'yellowReds', 'goals',
    'victories', 'ties', 'defeats',
    'leagueCountry', 'position'
]

X = pd.get_dummies(df[['dark'] + control_cols], drop_first=True)
X = sm.add_constant(X)

# OLS regression on redCards (continuous)

ols_model = sm.OLS(df['redCards'], X).fit()

# Logistic regression on any red card

logit_model = sm.Logit(df['any_red'], X).fit(disp=False)

# Save key results for downstream use / inspection

results = {
    'median_skin_tone': float(median_skin),
    'mean_red_light': float(mean_red_by_group.loc[0]),
    'mean_red_dark': float(mean_red_by_group.loc[1]),
    'rate_any_red_light': float(rate_any_red_by_group.loc[0]),
    'rate_any_red_dark': float(rate_any_red_by_group.loc[1]),
    'ols_dark_coef': float(ols_model.params['dark']),
    'ols_dark_p': float(ols_model.pvalues['dark']),
    'logit_dark_coef': float(logit_model.params['dark']),
    'logit_dark_p': float(logit_model.pvalues['dark'])
}

print('Median skin tone:', results['median_skin_tone'])
print('Mean redCards (light):', results['mean_red_light'])
print('Mean redCards (dark):', results['mean_red_dark'])
print('Any red rate (light):', results['rate_any_red_light'])
print('Any red rate (dark):', results['rate_any_red_dark'])
print('OLS dark coef:', results['ols_dark_coef'], 'p=', results['ols_dark_p'])
print('Logit dark coef:', results['logit_dark_coef'], 'p=', results['logit_dark_p'])

# Simple decision rule for conclusion

is_dark_higher = (
    results['mean_red_dark'] > results['mean_red_light']
    and results['rate_any_red_dark'] > results['rate_any_red_light']
    and results['ols_dark_coef'] > 0
    and results['logit_dark_coef'] > 0
)

with open('analysis_results.txt', 'w') as f:
    for k, v in results.items():
        f.write(f'{k}: {v}\n')
    f.write(f'is_dark_higher: {is_dark_higher}\n')
