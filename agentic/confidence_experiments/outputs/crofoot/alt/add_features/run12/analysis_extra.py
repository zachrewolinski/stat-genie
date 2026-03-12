import json
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')
cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
df = df[cols].dropna().copy()

# Relative size metrics
with np.errstate(divide='ignore', invalid='ignore'):
    df['size_ratio'] = df['n_focal'] / df['n_other']
    df['log_size_ratio'] = np.log(df['size_ratio'])

# Location metrics
with np.errstate(divide='ignore', invalid='ignore'):
    df['loc_ratio'] = df['dist_other'] / df['dist_focal']
    df['log_loc_ratio'] = np.log(df['loc_ratio'])

df['loc_closer_focal'] = (df['dist_focal'] < df['dist_other']).astype(int)
df['size_focal_larger'] = (df['n_focal'] > df['n_other']).astype(int)


def logit_summary(x_cols):
    X = df[x_cols].copy()
    for c in x_cols:
        if X[c].std(ddof=0) > 0:
            X[c] = (X[c] - X[c].mean()) / X[c].std(ddof=0)
    X = sm.add_constant(X)
    y = df['win']
    res = sm.Logit(y, X).fit(disp=False)
    return {
        'coef': res.params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
        'odds_ratio': np.exp(res.params).to_dict(),
        'odds_ratio_ci': {k: [float(v) for v in np.exp(res.conf_int()).loc[k]] for k in res.params.index},
    }

results = {
    'n_rows': len(df),
    'win_rate_overall': df['win'].mean(),
    'win_rate_by_size_larger': df.groupby('size_focal_larger')['win'].mean().to_dict(),
    'win_rate_by_loc_closer': df.groupby('loc_closer_focal')['win'].mean().to_dict(),
    'model_log_ratio': logit_summary(['log_size_ratio', 'log_loc_ratio']),
    'model_binary': logit_summary(['size_focal_larger', 'loc_closer_focal'])
}

print(json.dumps(results, indent=2))
