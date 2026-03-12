import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'crofoot.csv'

df = pd.read_csv(path)

# Relative group size and contest location
# Positive rel_size => focal larger; positive rel_dist => contest closer to focal center

df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_other'] - df['dist_focal']

# Alternative ratios (log) to reduce scale dependence
# Add a small epsilon to avoid division issues (not needed here but safe)

eps = 1e-9

df['log_size_ratio'] = np.log((df['n_focal'] + eps) / (df['n_other'] + eps))
df['log_dist_ratio'] = np.log((df['dist_other'] + eps) / (df['dist_focal'] + eps))


def fit_logit(cols):
    X = df[cols]
    X = sm.add_constant(X)
    y = df['win']
    model = sm.Logit(y, X)
    res = model.fit(disp=False)
    params = res.params
    conf = res.conf_int()
    odds = np.exp(params)
    odds_ci = np.exp(conf)
    return {
        'coef': params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
        'odds_ratio': odds.to_dict(),
        'odds_ratio_ci': odds_ci.to_dict(),
        'aic': res.aic,
        'pseudo_r2': res.prsquared,
    }

result = {
    'n': len(df),
    'model_diff': fit_logit(['rel_size', 'rel_dist']),
    'model_log_ratio': fit_logit(['log_size_ratio', 'log_dist_ratio']),
    'win_rate_rel_size_pos': df.groupby(df['rel_size'] > 0)['win'].mean().to_dict(),
    'win_rate_rel_dist_pos': df.groupby(df['rel_dist'] > 0)['win'].mean().to_dict(),
}

print(json.dumps(result, indent=2))
