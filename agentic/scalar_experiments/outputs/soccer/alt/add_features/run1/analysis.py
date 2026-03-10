import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('soccer.csv')

# Compute mean skin tone from raters
_df['skin_mean'] = _df[['rater1', 'rater2']].mean(axis=1)

# Filter to rows with needed data
_df = _df.dropna(subset=['skin_mean', 'redCards', 'games'])
_df = _df[_df['games'] > 0]

# Define dark vs light (scale 0-1, midpoint 0.5)
_df['dark'] = (_df['skin_mean'] > 0.5).astype(int)

# Group statistics
_group = _df.groupby('dark').agg(
    redCards_sum=('redCards', 'sum'),
    games_sum=('games', 'sum'),
    n_rows=('redCards', 'size')
).reset_index()
_group['rate_per_game'] = _group['redCards_sum'] / _group['games_sum']

# Poisson regression with offset for games
X = sm.add_constant(_df['dark'])
model = sm.GLM(_df['redCards'], X, family=sm.families.Poisson(), offset=np.log(_df['games']))
res = model.fit(cov_type='HC0')
coef = res.params['dark']
se = res.bse['dark']
pval = res.pvalues['dark']
irr = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Negative binomial (fixed alpha=1) robustness
nb_model = sm.GLM(_df['redCards'], X, family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(_df['games']))
nb_res = nb_model.fit(cov_type='HC0')
nb_coef = nb_res.params['dark']
nb_se = nb_res.bse['dark']
nb_pval = nb_res.pvalues['dark']
nb_irr = float(np.exp(nb_coef))
nb_ci_low = float(np.exp(nb_coef - 1.96 * nb_se))
nb_ci_high = float(np.exp(nb_coef + 1.96 * nb_se))

# Also test continuous skin_mean
Xc = sm.add_constant(_df['skin_mean'])
cont_model = sm.GLM(_df['redCards'], Xc, family=sm.families.Poisson(), offset=np.log(_df['games']))
cont_res = cont_model.fit(cov_type='HC0')
cont_coef = cont_res.params['skin_mean']
cont_se = cont_res.bse['skin_mean']
cont_pval = cont_res.pvalues['skin_mean']
cont_irr = float(np.exp(cont_coef))
cont_ci_low = float(np.exp(cont_coef - 1.96 * cont_se))
cont_ci_high = float(np.exp(cont_coef + 1.96 * cont_se))

summary = {
    'rows_used': int(_df.shape[0]),
    'group_stats': _group.to_dict(orient='records'),
    'poisson': {
        'coef': float(coef),
        'irr': irr,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'pval': float(pval),
    },
    'neg_bin': {
        'coef': float(nb_coef),
        'irr': nb_irr,
        'ci_low': nb_ci_low,
        'ci_high': nb_ci_high,
        'pval': float(nb_pval),
    },
    'poisson_continuous': {
        'coef': float(cont_coef),
        'irr': cont_irr,
        'ci_low': cont_ci_low,
        'ci_high': cont_ci_high,
        'pval': float(cont_pval),
    },
}

print(json.dumps(summary, indent=2))
