import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Derived predictors
df['rel_size'] = df['feature7'] - df['feature8']
df['rel_loc'] = df['feature6'] - df['feature5']

# Standardize for effect size comparison
df['rel_size_z'] = (df['rel_size'] - df['rel_size'].mean()) / df['rel_size'].std(ddof=0)
df['rel_loc_z'] = (df['rel_loc'] - df['rel_loc'].mean()) / df['rel_loc'].std(ddof=0)

y = df['feature4']

def fit_logit(cols):
    X = sm.add_constant(df[cols])
    model = sm.Logit(y, X).fit(disp=False)
    return model

# Multivariate model
model_full = fit_logit(['rel_size', 'rel_loc'])

# Univariate models
model_size = fit_logit(['rel_size'])
model_loc = fit_logit(['rel_loc'])

# Standardized full model
model_full_z = fit_logit(['rel_size_z', 'rel_loc_z'])

def summarize(model, label):
    params = model.params
    pvalues = model.pvalues
    summary = {
        'label': label,
        'n': int(model.nobs),
        'params': params.to_dict(),
        'pvalues': pvalues.to_dict(),
        'llf': float(model.llf),
        'aic': float(model.aic),
        'pseudo_r2': float(model.prsquared),
    }
    return summary

out = {
    'full': summarize(model_full, 'rel_size + rel_loc'),
    'size_only': summarize(model_size, 'rel_size'),
    'loc_only': summarize(model_loc, 'rel_loc'),
    'full_z': summarize(model_full_z, 'rel_size_z + rel_loc_z'),
    'win_rate_by_rel_size_sign': df.assign(
        rel_size_sign=np.where(df['rel_size'] > 0, 'focal_larger',
                               np.where(df['rel_size'] < 0, 'focal_smaller', 'equal_size'))
    ).groupby('rel_size_sign')['feature4'].agg(['mean', 'count']).to_dict(),
    'win_rate_by_rel_loc_sign': df.assign(
        rel_loc_sign=np.where(df['rel_loc'] > 0, 'focal_closer_to_center',
                              np.where(df['rel_loc'] < 0, 'other_closer_to_center', 'equal_distance'))
    ).groupby('rel_loc_sign')['feature4'].agg(['mean', 'count']).to_dict(),
}

print(json.dumps(out, indent=2, sort_keys=True))
