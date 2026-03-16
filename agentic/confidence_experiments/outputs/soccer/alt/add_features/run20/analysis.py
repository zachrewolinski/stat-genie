import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('soccer.csv')

# Build skin tone measure (mean of raters)
df['skin'] = df[['rater1', 'rater2']].mean(axis=1)

# Filter valid rows
df = df[(df['skin'].notna()) & (df['games'] > 0) & (df['redCards'].notna())]

# Basic stats
n_rows = len(df)

# Poisson regression for red card counts with exposure (games)
X = sm.add_constant(df['skin'])
offset = np.log(df['games'])

poisson_model = sm.GLM(df['redCards'], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type='HC3')

beta = poisson_res.params['skin']
se = poisson_res.bse['skin']
pval = poisson_res.pvalues['skin']
irr = float(np.exp(beta))

# Predicted per-game rate for skin=0 (very light) and skin=1 (very dark)
# Using games=1 so rate per game
pred_light = float(poisson_res.predict([1, 0], offset=0))
pred_dark = float(poisson_res.predict([1, 1], offset=0))

# Grouped comparison: light vs dark using thresholds
df['skin_group'] = pd.cut(
    df['skin'],
    bins=[-0.001, 0.25, 0.75, 1.001],
    labels=['light', 'medium', 'dark']
)

group_summary = (
    df.groupby('skin_group', observed=True)
      .agg(redCards=('redCards', 'sum'), games=('games', 'sum'), dyads=('redCards', 'size'))
      .assign(rate_per_game=lambda d: d['redCards'] / d['games'])
)

# Simple rate ratio dark vs light
if 'dark' in group_summary.index and 'light' in group_summary.index:
    dark_rate = float(group_summary.loc['dark', 'rate_per_game'])
    light_rate = float(group_summary.loc['light', 'rate_per_game'])
    rate_ratio = dark_rate / light_rate if light_rate > 0 else np.nan
else:
    dark_rate = light_rate = rate_ratio = np.nan

# Overdispersion check
overdispersion = float(poisson_res.deviance / poisson_res.df_resid) if poisson_res.df_resid > 0 else np.nan

results = {
    'n_rows': int(n_rows),
    'beta_skin': float(beta),
    'se_skin': float(se),
    'pval_skin': float(pval),
    'irr_skin': irr,
    'pred_rate_light': pred_light,
    'pred_rate_dark': pred_dark,
    'group_rates': group_summary.reset_index().to_dict(orient='records'),
    'dark_rate': dark_rate,
    'light_rate': light_rate,
    'rate_ratio_dark_vs_light': rate_ratio,
    'overdispersion': overdispersion,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
