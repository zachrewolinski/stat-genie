import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

_df = pd.read_csv('soccer.csv')
_df['skin_mean'] = _df[['rater1', 'rater2']].mean(axis=1)
_df = _df.dropna(subset=['skin_mean', 'redCards', 'games'])
_df = _df[_df['games'] > 0]

# Extreme groups
_df['group'] = np.where(_df['skin_mean'] <= 0.25, 'light', np.where(_df['skin_mean'] >= 0.75, 'dark', 'mid'))
_df_ext = _df[_df['group'].isin(['light', 'dark'])].copy()
_df_ext['dark'] = (_df_ext['group'] == 'dark').astype(int)

_group = _df_ext.groupby('group').agg(redCards_sum=('redCards','sum'), games_sum=('games','sum'), n_rows=('redCards','size')).reset_index()
_group['rate_per_game'] = _group['redCards_sum'] / _group['games_sum']

X = sm.add_constant(_df_ext['dark'])
model = sm.GLM(_df_ext['redCards'], X, family=sm.families.Poisson(), offset=np.log(_df_ext['games']))
res = model.fit(cov_type='HC0')
coef = res.params['dark']
se = res.bse['dark']

summary = {
    'rows_used': int(_df_ext.shape[0]),
    'group_stats': _group.to_dict(orient='records'),
    'poisson': {
        'coef': float(coef),
        'irr': float(np.exp(coef)),
        'ci_low': float(np.exp(coef - 1.96 * se)),
        'ci_high': float(np.exp(coef + 1.96 * se)),
        'pval': float(res.pvalues['dark']),
    },
}

print(json.dumps(summary, indent=2))
