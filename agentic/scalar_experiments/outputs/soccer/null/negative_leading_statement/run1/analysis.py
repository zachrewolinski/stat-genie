import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('soccer.csv')

# Average skin tone from two raters
_df['skin_mean'] = _df[['rater1', 'rater2']].mean(axis=1)

# Keep rows with skin info and positive games
_df = _df[_df['skin_mean'].notna()].copy()
_df = _df[_df['games'] > 0].copy()

# Define dark vs light based on midpoint 0.5 (scale 0-1)
_df['skin_dark'] = (_df['skin_mean'] > 0.5).astype(int)

# Red cards per game
_df['red_per_game'] = _df['redCards'] / _df['games']

# Basic group stats
summary = _df.groupby('skin_dark').agg(
    n=('redCards', 'size'),
    players=('playerShort', 'nunique'),
    total_games=('games', 'sum'),
    total_red=('redCards', 'sum'),
    mean_red_per_game=('red_per_game', 'mean')
).reset_index()

# Poisson regression with offset for games
# Use skin_mean continuous and skin_dark binary
# Include position and league as categorical controls (drop NA)
model_df = _df.dropna(subset=['position', 'leagueCountry']).copy()

# Poisson with skin_mean
poisson_cont = smf.glm(
    formula='redCards ~ skin_mean + C(position) + C(leagueCountry)',
    data=model_df,
    family=sm.families.Poisson(),
    offset=np.log(model_df['games'])
).fit(cov_type='HC1')

# Poisson with skin_dark
poisson_bin = smf.glm(
    formula='redCards ~ skin_dark + C(position) + C(leagueCountry)',
    data=model_df,
    family=sm.families.Poisson(),
    offset=np.log(model_df['games'])
).fit(cov_type='HC1')

# Overdispersion check (Pearson chi2 / df)
pearson_disp = poisson_cont.pearson_chi2 / poisson_cont.df_resid

# If overdispersed, fit Negative Binomial (NB2)
nb_cont = smf.glm(
    formula='redCards ~ skin_mean + C(position) + C(leagueCountry)',
    data=model_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(model_df['games'])
).fit(cov_type='HC1')

nb_bin = smf.glm(
    formula='redCards ~ skin_dark + C(position) + C(leagueCountry)',
    data=model_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(model_df['games'])
).fit(cov_type='HC1')

# Extract effects
results = {
    'summary': summary,
    'poisson_cont': {
        'coef': float(poisson_cont.params['skin_mean']),
        'se': float(poisson_cont.bse['skin_mean']),
        'pvalue': float(poisson_cont.pvalues['skin_mean']),
        'ir': float(np.exp(poisson_cont.params['skin_mean']))
    },
    'poisson_bin': {
        'coef': float(poisson_bin.params['skin_dark']),
        'se': float(poisson_bin.bse['skin_dark']),
        'pvalue': float(poisson_bin.pvalues['skin_dark']),
        'ir': float(np.exp(poisson_bin.params['skin_dark']))
    },
    'nb_cont': {
        'coef': float(nb_cont.params['skin_mean']),
        'se': float(nb_cont.bse['skin_mean']),
        'pvalue': float(nb_cont.pvalues['skin_mean']),
        'ir': float(np.exp(nb_cont.params['skin_mean']))
    },
    'nb_bin': {
        'coef': float(nb_bin.params['skin_dark']),
        'se': float(nb_bin.bse['skin_dark']),
        'pvalue': float(nb_bin.pvalues['skin_dark']),
        'ir': float(np.exp(nb_bin.params['skin_dark']))
    },
    'pearson_dispersion': float(pearson_disp)
}

# Save intermediate results
results_out = {
    'summary': results['summary'].to_dict(orient='records'),
    'poisson_cont': results['poisson_cont'],
    'poisson_bin': results['poisson_bin'],
    'nb_cont': results['nb_cont'],
    'nb_bin': results['nb_bin'],
    'pearson_dispersion': results['pearson_dispersion'],
}

with open('analysis_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)

print(json.dumps(results_out, indent=2))
