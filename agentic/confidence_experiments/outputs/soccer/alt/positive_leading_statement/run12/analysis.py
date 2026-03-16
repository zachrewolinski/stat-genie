import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Create skin tone measure as mean of raters
# If one rater missing, use the other
skin = df[['rater1', 'rater2']].mean(axis=1, skipna=True)
df = df.copy()
df['skin_tone'] = skin

# Drop rows with no skin rating or no games or redCards missing
analysis_df = df.dropna(subset=['skin_tone', 'games', 'redCards'])
analysis_df = analysis_df[analysis_df['games'] > 0]

# Basic descriptives
summary = {
    'n_total': int(len(df)),
    'n_with_skin': int(df['skin_tone'].notna().sum()),
    'n_analysis': int(len(analysis_df)),
    'redcards_mean': float(analysis_df['redCards'].mean()),
    'redcards_rate_per_game_mean': float((analysis_df['redCards'] / analysis_df['games']).mean()),
}

# Compare light vs dark (split at median skin tone)
median_skin = analysis_df['skin_tone'].median()
analysis_df['skin_group'] = np.where(analysis_df['skin_tone'] >= median_skin, 'darker', 'lighter')

# Aggregate red card rates by skin group
rate_by_group = (
    analysis_df.groupby('skin_group')
    .apply(lambda g: pd.Series({
        'n': len(g),
        'redcards_total': g['redCards'].sum(),
        'games_total': g['games'].sum(),
        'redcards_per_game': g['redCards'].sum() / g['games'].sum()
    }))
    .reset_index()
)

# Poisson regression with log(games) offset
# Model: redCards ~ skin_tone + position + leagueCountry
# Include controls to reduce confounding
analysis_df['log_games'] = np.log(analysis_df['games'])

# Some categories may have missing values; drop for regression
reg_df = analysis_df.dropna(subset=['position', 'leagueCountry'])

poisson_model = smf.glm(
    formula='redCards ~ skin_tone + C(position) + C(leagueCountry)',
    data=reg_df,
    family=sm.families.Poisson(),
    offset=reg_df['log_games']
).fit(cov_type='HC0')

# Check overdispersion; compute robust standard errors in summary (HC0) already

# Also run a simple unadjusted Poisson with only skin_tone
poisson_simple = smf.glm(
    formula='redCards ~ skin_tone',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
).fit(cov_type='HC0')

# Logistic regression on any red card (binary), with games as covariate and controls
analysis_df['any_red'] = (analysis_df['redCards'] > 0).astype(int)
logit_df = analysis_df.dropna(subset=['position', 'leagueCountry'])
logit_model = smf.logit(
    formula='any_red ~ skin_tone + games + C(position) + C(leagueCountry)',
    data=logit_df
).fit(disp=False)

# Extract key results
results = {
    'median_skin': float(median_skin),
    'rate_by_group': rate_by_group.to_dict(orient='records'),
    'poisson_simple': {
        'coef_skin': float(poisson_simple.params['skin_tone']),
        'se_skin': float(poisson_simple.bse['skin_tone']),
        'p_value_skin': float(poisson_simple.pvalues['skin_tone']),
        'irr_skin': float(np.exp(poisson_simple.params['skin_tone'])),
        'ci_low': float(np.exp(poisson_simple.conf_int().loc['skin_tone'][0])),
        'ci_high': float(np.exp(poisson_simple.conf_int().loc['skin_tone'][1])),
    },
    'poisson_adjusted': {
        'coef_skin': float(poisson_model.params['skin_tone']),
        'se_skin': float(poisson_model.bse['skin_tone']),
        'p_value_skin': float(poisson_model.pvalues['skin_tone']),
        'irr_skin': float(np.exp(poisson_model.params['skin_tone'])),
        'ci_low': float(np.exp(poisson_model.conf_int().loc['skin_tone'][0])),
        'ci_high': float(np.exp(poisson_model.conf_int().loc['skin_tone'][1])),
    },
    'logit_adjusted': {
        'coef_skin': float(logit_model.params['skin_tone']),
        'se_skin': float(logit_model.bse['skin_tone']),
        'p_value_skin': float(logit_model.pvalues['skin_tone']),
        'or_skin': float(np.exp(logit_model.params['skin_tone'])),
        'ci_low': float(np.exp(logit_model.conf_int().loc['skin_tone'][0])),
        'ci_high': float(np.exp(logit_model.conf_int().loc['skin_tone'][1])),
    },
}

with open('analysis_results.json', 'w') as f:
    json.dump({'summary': summary, 'results': results}, f, indent=2)

print(json.dumps({'summary': summary, 'results': results}, indent=2))
