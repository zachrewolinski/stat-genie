import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
file_path = 'crofoot.csv'
df = pd.read_csv(file_path)

# Relevant columns
cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
df_sub = df[cols].copy()

# Drop missing
before = len(df_sub)
df_sub = df_sub.dropna()

# Construct predictors
# Relative group size (difference)
df_sub['size_diff'] = df_sub['n_focal'] - df_sub['n_other']
# Relative location advantage: positive if contest closer to focal (other farther from its center)
df_sub['loc_adv'] = df_sub['dist_other'] - df_sub['dist_focal']

# Standardize predictors for comparable coefficients
for col in ['size_diff', 'loc_adv']:
    df_sub[f'z_{col}'] = (df_sub[col] - df_sub[col].mean()) / df_sub[col].std(ddof=0)

# Logistic regression: win ~ size_diff + loc_adv
X = df_sub[['z_size_diff', 'z_loc_adv']]
X = sm.add_constant(X)
y = df_sub['win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Odds ratios for 1 SD increase
params = result.params
conf = result.conf_int()

or_ = np.exp(params)
or_ci = np.exp(conf)

summary = {
    'n_rows': len(df_sub),
    'dropped_rows': before - len(df_sub),
    'mean_win': df_sub['win'].mean(),
    'coef': params.to_dict(),
    'pvalues': result.pvalues.to_dict(),
    'odds_ratio': or_.to_dict(),
    'odds_ratio_ci': {k: [or_ci.loc[k, 0], or_ci.loc[k, 1]] for k in or_ci.index},
}

# Also run single-predictor models for sensitivity
models = {}
for col in ['z_size_diff', 'z_loc_adv']:
    X1 = sm.add_constant(df_sub[[col]])
    res1 = sm.Logit(y, X1).fit(disp=False)
    models[col] = {
        'coef': res1.params.to_dict(),
        'pvalues': res1.pvalues.to_dict(),
        'odds_ratio': np.exp(res1.params).to_dict(),
        'odds_ratio_ci': {k: [float(v) for v in np.exp(res1.conf_int()).loc[k]] for k in res1.params.index},
    }

summary['single_predictor_models'] = models

print(json.dumps(summary, indent=2))
