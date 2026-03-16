import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data

df = pd.read_csv('soccer.csv')

# Skin tone average
skin = df[['rater1', 'rater2']].mean(axis=1)

df = df.assign(skin=skin)

# Keep rows with needed fields
needed = ['skin', 'redCards', 'games', 'playerShort']
df = df.dropna(subset=needed)
# Guard against non-positive games

df = df[df['games'] > 0]

# Basic counts
n_total = len(df)

# Define light/dark groups using the midpoint as neutral (0.5)
light_mask = df['skin'] < 0.5
dark_mask = df['skin'] >= 0.5

# Also define extreme terciles for robustness
q_low, q_high = df['skin'].quantile([0.33, 0.67])
light_terc = df['skin'] <= q_low
dark_terc = df['skin'] >= q_high

# Compute rates per 100 games

def rate_per_100(mask):
    red = df.loc[mask, 'redCards'].sum()
    games = df.loc[mask, 'games'].sum()
    rate = (red / games) * 100 if games > 0 else np.nan
    return red, games, rate


light_red, light_games, light_rate = rate_per_100(light_mask)
dark_red, dark_games, dark_rate = rate_per_100(dark_mask)

light_red_t, light_games_t, light_rate_t = rate_per_100(light_terc)
dark_red_t, dark_games_t, dark_rate_t = rate_per_100(dark_terc)

# Poisson regression with exposure offset
X = sm.add_constant(df['skin'])
y = df['redCards']
offset = np.log(df['games'])

model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
result = model.fit(cov_type='cluster', cov_kwds={'groups': df['playerShort']})

beta = result.params['skin']
se = result.bse['skin']
pval = result.pvalues['skin']

# Incidence rate ratio for full scale (0 to 1)
irr_full = float(np.exp(beta))
ci_low_full = float(np.exp(beta - 1.96 * se))
ci_high_full = float(np.exp(beta + 1.96 * se))

# Predicted dark vs light (0.75 vs 0.25) rate ratio
delta = 0.75 - 0.25
irr_dl = float(np.exp(beta * delta))
ci_low_dl = float(np.exp((beta - 1.96 * se) * delta))
ci_high_dl = float(np.exp((beta + 1.96 * se) * delta))

out = {
    'n_total': int(n_total),
    'light': {
        'n': int(light_mask.sum()),
        'red_cards': float(light_red),
        'games': float(light_games),
        'rate_per_100_games': float(light_rate),
    },
    'dark': {
        'n': int(dark_mask.sum()),
        'red_cards': float(dark_red),
        'games': float(dark_games),
        'rate_per_100_games': float(dark_rate),
    },
    'light_tercile': {
        'n': int(light_terc.sum()),
        'rate_per_100_games': float(light_rate_t),
    },
    'dark_tercile': {
        'n': int(dark_terc.sum()),
        'rate_per_100_games': float(dark_rate_t),
    },
    'poisson': {
        'beta_skin': float(beta),
        'se_skin': float(se),
        'pvalue_skin': float(pval),
        'irr_full_scale': irr_full,
        'irr_full_scale_ci': [ci_low_full, ci_high_full],
        'irr_dark_vs_light_0_75_0_25': irr_dl,
        'irr_dark_vs_light_ci': [ci_low_dl, ci_high_dl],
    },
}

print(json.dumps(out, indent=2))
