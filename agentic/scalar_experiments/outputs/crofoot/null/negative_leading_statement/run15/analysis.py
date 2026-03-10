import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError


df = pd.read_csv('crofoot.csv')

# Derived predictors
# Relative size: focal - other (positive means focal larger)
# Relative distance: other - focal (positive means contest is closer to focal's home range center)

df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_other'] - df['dist_focal']


def fit_logit(y, X):
    X = sm.add_constant(X, has_constant='add')
    try:
        model = sm.Logit(y, X).fit(disp=False)
    except PerfectSeparationError:
        return None
    return model


def summarize_model(model, name):
    if model is None:
        return {
            'name': name,
            'converged': False,
            'error': 'perfect separation'
        }
    params = model.params
    conf = model.conf_int()
    out = {
        'name': name,
        'converged': model.mle_retvals.get('converged', True),
        'n': int(model.nobs),
        'llf': model.llf,
        'llnull': model.llnull,
        'pseudo_r2': 1 - model.llf / model.llnull,
        'coef': params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'odds_ratio': np.exp(params).to_dict(),
        'or_ci_low': np.exp(conf[0]).to_dict(),
        'or_ci_high': np.exp(conf[1]).to_dict(),
    }
    return out


results = {}

# Model A: rel_size + rel_dist
model_a = fit_logit(df['win'], df[['rel_size', 'rel_dist']])
results['model_a'] = summarize_model(model_a, 'win ~ rel_size + rel_dist')

# Model B: rel_size only
model_b = fit_logit(df['win'], df[['rel_size']])
results['model_b'] = summarize_model(model_b, 'win ~ rel_size')

# Model C: rel_dist only
model_c = fit_logit(df['win'], df[['rel_dist']])
results['model_c'] = summarize_model(model_c, 'win ~ rel_dist')

# Model D: separate focal/other size and distance
model_d = fit_logit(df['win'], df[['n_focal', 'n_other', 'dist_focal', 'dist_other']])
results['model_d'] = summarize_model(model_d, 'win ~ n_focal + n_other + dist_focal + dist_other')

# Simple descriptive stats
summary = {
    'n': int(len(df)),
    'win_rate': float(df['win'].mean()),
    'rel_size_mean': float(df['rel_size'].mean()),
    'rel_size_std': float(df['rel_size'].std()),
    'rel_dist_mean': float(df['rel_dist'].mean()),
    'rel_dist_std': float(df['rel_dist'].std()),
}

print('SUMMARY')
print(summary)
print('\nMODELS')
for k, v in results.items():
    print('\n', k)
    for kk, vv in v.items():
        print(f"{kk}: {vv}")
