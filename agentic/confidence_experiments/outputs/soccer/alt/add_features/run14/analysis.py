import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path('soccer.csv')

df = pd.read_csv(DATA_PATH)

# Compute average skin tone rating
skin = df[['rater1', 'rater2']].mean(axis=1)
df = df.copy()
df['skin_avg'] = skin

# Drop rows without skin tone or games/redCards
analysis_df = df.dropna(subset=['skin_avg', 'games', 'redCards']).copy()

# Create binary dark vs light using scale-based thresholds
# Skin ratings are on a 5-point scale normalized to [0, 1] with steps of 0.25.
# Define light as <= 0.25 (very light/light) and dark as >= 0.75 (dark/very dark).
light_thresh = 0.25
dark_thresh = 0.75
analysis_df['skin_group'] = np.where(analysis_df['skin_avg'] >= dark_thresh, 'dark',
                                   np.where(analysis_df['skin_avg'] <= light_thresh, 'light', 'mid'))

# Binary for any red card in dyad
analysis_df['any_red'] = (analysis_df['redCards'] > 0).astype(int)

# Summaries
summary = {}
summary['n_rows'] = len(analysis_df)
summary['light_thresh'] = float(light_thresh)
summary['dark_thresh'] = float(dark_thresh)
summary['group_counts'] = analysis_df['skin_group'].value_counts().to_dict()

# Red card rate per game by group
rate_by_group = (analysis_df.groupby('skin_group')
                 .apply(lambda g: g['redCards'].sum() / g['games'].sum()))
summary['rate_per_game'] = rate_by_group.to_dict()

# Logistic regression: any red card ~ skin_avg + log(games)
# Using log(games) as exposure control.
analysis_df['log_games'] = np.log(analysis_df['games'])
logit_model = smf.logit('any_red ~ skin_avg + log_games', data=analysis_df).fit(disp=0)
summary['logit'] = {
    'coef_skin_avg': float(logit_model.params['skin_avg']),
    'p_skin_avg': float(logit_model.pvalues['skin_avg']),
    'odds_ratio_skin_avg': float(np.exp(logit_model.params['skin_avg']))
}

# Poisson regression for count with offset log(games)
# Ensure games > 0
poisson_df = analysis_df[analysis_df['games'] > 0].copy()
poisson_model = smf.glm('redCards ~ skin_avg', data=poisson_df,
                        family=sm.families.Poisson(),
                        offset=np.log(poisson_df['games'])).fit()
summary['poisson'] = {
    'coef_skin_avg': float(poisson_model.params['skin_avg']),
    'p_skin_avg': float(poisson_model.pvalues['skin_avg']),
    'rate_ratio_skin_avg': float(np.exp(poisson_model.params['skin_avg']))
}

# Compare dark vs light using Poisson with group indicator
binary_df = analysis_df[analysis_df['skin_group'].isin(['dark', 'light'])].copy()
binary_df['dark'] = (binary_df['skin_group'] == 'dark').astype(int)
poisson_bin = smf.glm('redCards ~ dark', data=binary_df,
                      family=sm.families.Poisson(),
                      offset=np.log(binary_df['games'])).fit()
summary['poisson_dark_vs_light'] = {
    'coef_dark': float(poisson_bin.params['dark']),
    'p_dark': float(poisson_bin.pvalues['dark']),
    'rate_ratio_dark': float(np.exp(poisson_bin.params['dark']))
}

# Logistic dark vs light
logit_bin = smf.logit('any_red ~ dark + log_games', data=binary_df).fit(disp=0)
summary['logit_dark_vs_light'] = {
    'coef_dark': float(logit_bin.params['dark']),
    'p_dark': float(logit_bin.pvalues['dark']),
    'odds_ratio_dark': float(np.exp(logit_bin.params['dark']))
}

# Save summary
out = Path('analysis_summary.json')
import json
with out.open('w') as f:
    json.dump(summary, f, indent=2)

print('Wrote analysis_summary.json')
