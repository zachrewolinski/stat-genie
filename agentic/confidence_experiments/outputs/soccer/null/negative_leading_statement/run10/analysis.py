import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'soccer.csv'
df = pd.read_csv(path)

# Compute mean skin tone from rater1 and rater2
# Some rows may have NaN if no photo

df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Keep rows with skin tone and games > 0
analysis_df = df[(df['skin_tone'].notna()) & (df['games'] > 0)].copy()

# Create red card per game rate for descriptive stats
analysis_df['red_per_game'] = analysis_df['redCards'] / analysis_df['games']

# Skin tone bins: light (<=0.25), medium (0.5), dark (>=0.75) on normalized 0-1 scale
# Use quantiles to define light/dark groups for robust comparisons
analysis_df['skin_quantile'] = pd.qcut(analysis_df['skin_tone'], q=4, labels=['Q1_light','Q2','Q3','Q4_dark'])

# Summary stats by quantile
summary = analysis_df.groupby('skin_quantile').agg(
    n=('redCards','size'),
    total_games=('games','sum'),
    total_red=('redCards','sum'),
    red_rate=('red_per_game','mean')
).reset_index()
summary['red_per_game_total'] = summary['total_red'] / summary['total_games']

# Poisson regression with log(games) offset
# This models expected redCards per game as function of skin_tone
analysis_df['log_games'] = np.log(analysis_df['games'])

# add controls: position, leagueCountry, and player fixed effects? too heavy. We'll include position and league country.
# Also include yellow cards as proxy for aggressiveness (caution). But could be mediator. We'll check base and extended.

# Base model
model_base = smf.glm(
    formula='redCards ~ skin_tone',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
).fit()

# Extended model with controls
model_ctrl = smf.glm(
    formula='redCards ~ skin_tone + yellowCards + position + leagueCountry',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
).fit()

# Binary outcome: any red card in dyad
analysis_df['any_red'] = (analysis_df['redCards'] > 0).astype(int)
logit_base = smf.logit('any_red ~ skin_tone', data=analysis_df).fit(disp=False)
logit_ctrl = smf.logit('any_red ~ skin_tone + yellowCards + position + leagueCountry', data=analysis_df).fit(disp=False)

# Compute effect sizes
base_coef = model_base.params['skin_tone']
base_p = model_base.pvalues['skin_tone']
base_ir = np.exp(base_coef)

ctrl_coef = model_ctrl.params['skin_tone']
ctrl_p = model_ctrl.pvalues['skin_tone']
ctrl_ir = np.exp(ctrl_coef)

logit_base_coef = logit_base.params['skin_tone']
logit_base_p = logit_base.pvalues['skin_tone']
logit_base_or = np.exp(logit_base_coef)

logit_ctrl_coef = logit_ctrl.params['skin_tone']
logit_ctrl_p = logit_ctrl.pvalues['skin_tone']
logit_ctrl_or = np.exp(logit_ctrl_coef)

# Summaries for light vs dark quantiles
q1 = summary.loc[summary['skin_quantile']=='Q1_light'].iloc[0]
q4 = summary.loc[summary['skin_quantile']=='Q4_dark'].iloc[0]

results = {
    'rows_total': len(df),
    'rows_with_skin': len(analysis_df),
    'summary_by_quantile': summary,
    'poisson_base': {'coef': base_coef, 'p': base_p, 'ir': base_ir},
    'poisson_ctrl': {'coef': ctrl_coef, 'p': ctrl_p, 'ir': ctrl_ir},
    'logit_base': {'coef': logit_base_coef, 'p': logit_base_p, 'or': logit_base_or},
    'logit_ctrl': {'coef': logit_ctrl_coef, 'p': logit_ctrl_p, 'or': logit_ctrl_or},
    'q1': q1.to_dict(),
    'q4': q4.to_dict(),
}

# Print results
print('Rows total:', results['rows_total'])
print('Rows with skin:', results['rows_with_skin'])
print('\nSummary by skin tone quartile:')
print(summary)
print('\nPoisson base:', results['poisson_base'])
print('Poisson ctrl:', results['poisson_ctrl'])
print('Logit base:', results['logit_base'])
print('Logit ctrl:', results['logit_ctrl'])
print('\nQ1 light:', results['q1'])
print('Q4 dark:', results['q4'])
