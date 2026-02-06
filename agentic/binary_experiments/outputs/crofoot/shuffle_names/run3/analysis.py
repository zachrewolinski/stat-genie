import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
info = json.load(open('info.json', 'r'))
df = pd.read_csv('crofoot.csv')

# Map columns using descriptions in info.json
col_desc = {f['column']: f['properties']['description'] for f in info['data_desc']['fields']}

# Identify columns by their descriptions
# Outcome: 1 if focal won contest, 0 if other won
outcome_col = [c for c, d in col_desc.items() if 'focal won contest' in d][0]
# Distances to home range centers
focal_dist_col = [c for c, d in col_desc.items() if 'Distance in meters of focal group from the center of its home range' in d][0]
other_dist_col = [c for c, d in col_desc.items() if 'Distance in meters of other group from the center of its home range' in d][0]
# Group sizes (individual counts)
focal_size_col = [c for c, d in col_desc.items() if d == 'Number of individuals in focal group'][0]
other_size_col = [c for c, d in col_desc.items() if d == 'Number of individuals in other group'][0]

# Build analysis dataframe
analysis_df = df[[outcome_col, focal_dist_col, other_dist_col, focal_size_col, other_size_col]].copy()
analysis_df.rename(
    columns={
        outcome_col: 'win_focal',
        focal_dist_col: 'dist_focal_center',
        other_dist_col: 'dist_other_center',
        focal_size_col: 'size_focal',
        other_size_col: 'size_other',
    },
    inplace=True,
)

# Relative predictors
analysis_df['size_diff'] = analysis_df['size_focal'] - analysis_df['size_other']
analysis_df['location_adv'] = analysis_df['dist_other_center'] - analysis_df['dist_focal_center']
# Positive location_adv => focal is closer to its center than other is to its center

# Standardize predictors for interpretability
analysis_df['size_diff_z'] = (analysis_df['size_diff'] - analysis_df['size_diff'].mean()) / analysis_df['size_diff'].std()
analysis_df['location_adv_z'] = (analysis_df['location_adv'] - analysis_df['location_adv'].mean()) / analysis_df['location_adv'].std()

# Logistic regression
X = analysis_df[['size_diff_z', 'location_adv_z']]
X = sm.add_constant(X)
y = analysis_df['win_focal']

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Compute odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf.columns = ['ci_low', 'ci_high']

odds_ratios = np.exp(params)
conf_or = np.exp(conf)

summary = pd.DataFrame({
    'coef': params,
    'odds_ratio': odds_ratios,
    'ci_low_or': conf_or['ci_low'],
    'ci_high_or': conf_or['ci_high'],
    'p_value': result.pvalues,
})

# Save outputs for inspection
summary.to_csv('analysis_results.csv', index=True)

# Print a compact summary
print('Logistic regression on focal win (1=win)')
print(summary)

# Simple correlation checks
print('\nCorrelation between predictors and outcome:')
print(analysis_df[['win_focal','size_diff','location_adv']].corr())

# Convenience for decision
print('\nMean win rate by size advantage sign:')
print(analysis_df.groupby(analysis_df['size_diff']>0)['win_focal'].mean())
print('\nMean win rate by location advantage sign:')
print(analysis_df.groupby(analysis_df['location_adv']>0)['win_focal'].mean())
