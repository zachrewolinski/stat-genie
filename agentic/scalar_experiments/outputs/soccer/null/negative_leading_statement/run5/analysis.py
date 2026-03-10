import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('soccer.csv')

# Create skin tone average

df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Filter rows with skin tone and games > 0

df = df[df['skin_tone'].notna() & df['games'].notna() & (df['games'] > 0)].copy()

# Create binary skin tone group (light <= 0.5, dark > 0.5)

df['skin_group'] = np.where(df['skin_tone'] > 0.5, 'dark', 'light')

# Aggregate by skin group: total red cards and games

agg = df.groupby('skin_group').agg(
    red_cards=('redCards', 'sum'),
    games=('games', 'sum'),
    dyads=('redCards', 'size')
).reset_index()
agg['red_cards_per_game'] = agg['red_cards'] / agg['games']

# Simple rate ratio dark vs light

if set(agg['skin_group']) == {'dark', 'light'}:
    rate_dark = agg.loc[agg['skin_group'] == 'dark', 'red_cards_per_game'].iloc[0]
    rate_light = agg.loc[agg['skin_group'] == 'light', 'red_cards_per_game'].iloc[0]
    rate_ratio = rate_dark / rate_light if rate_light > 0 else np.nan
else:
    rate_dark = rate_light = rate_ratio = np.nan

# Poisson regression with offset log(games)

df['log_games'] = np.log(df['games'])

# Using continuous skin tone

model_cont = smf.glm(
    formula='redCards ~ skin_tone',
    data=df,
    family=sm.families.Poisson(),
    offset=df['log_games']
).fit(cov_type='HC0')

# Using binary skin group

df['skin_dark'] = (df['skin_group'] == 'dark').astype(int)

model_bin = smf.glm(
    formula='redCards ~ skin_dark',
    data=df,
    family=sm.families.Poisson(),
    offset=df['log_games']
).fit(cov_type='HC0')

# Logistic for any red card

df['any_red'] = (df['redCards'] > 0).astype(int)

logit = smf.glm(
    formula='any_red ~ skin_tone',
    data=df,
    family=sm.families.Binomial()
).fit(cov_type='HC0')

results = {
    'n_rows': int(len(df)),
    'rate_dark': float(rate_dark),
    'rate_light': float(rate_light),
    'rate_ratio': float(rate_ratio),
    'poisson_cont_coef': float(model_cont.params['skin_tone']),
    'poisson_cont_p': float(model_cont.pvalues['skin_tone']),
    'poisson_cont_ci': [float(x) for x in model_cont.conf_int().loc['skin_tone'].tolist()],
    'poisson_bin_coef': float(model_bin.params['skin_dark']),
    'poisson_bin_p': float(model_bin.pvalues['skin_dark']),
    'poisson_bin_ci': [float(x) for x in model_bin.conf_int().loc['skin_dark'].tolist()],
    'logit_coef': float(logit.params['skin_tone']),
    'logit_p': float(logit.pvalues['skin_tone']),
    'logit_ci': [float(x) for x in logit.conf_int().loc['skin_tone'].tolist()],
}

print(results)
