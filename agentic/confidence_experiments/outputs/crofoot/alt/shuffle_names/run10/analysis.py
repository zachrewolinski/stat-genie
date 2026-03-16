import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant

# Load data

df = pd.read_csv('crofoot.csv')

# Map columns based on info.json descriptions
# outcome: 1 if focal won contest, 0 if other won
outcome = 'm_focal'

# group sizes
focal_size = 'f_other'  # number of individuals in focal group
other_size = 'win'      # number of individuals in other group

# contest location (distance from each group's home-range center)
# smaller distance => closer to that group's center
focal_dist = 'm_other'
other_dist = 'n_focal'

# compute relative variables

df = df.copy()
df['size_diff'] = df[focal_size] - df[other_size]
# log ratio to reduce scale and handle symmetry
# add small epsilon to avoid division issues (not needed here, but safe)
df['size_log_ratio'] = np.log((df[focal_size] + 1e-9) / (df[other_size] + 1e-9))

# location advantage: positive means contest closer to focal group's center
# i.e., other_dist > focal_dist

df['loc_adv'] = df[other_dist] - df[focal_dist]

# Standardize predictors for interpretability (optional)

def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)

for col in ['size_diff', 'size_log_ratio', 'loc_adv']:
    df[f'z_{col}'] = zscore(df[col])

# Fit logistic regression with size difference and location advantage
X = df[['z_size_diff', 'z_loc_adv']]
X = add_constant(X)
y = df[outcome]

model = sm.Logit(y, X).fit(disp=False)

# Also fit model using log ratio instead of diff
X2 = df[['z_size_log_ratio', 'z_loc_adv']]
X2 = add_constant(X2)
model2 = sm.Logit(y, X2).fit(disp=False)

# Extract results

def summarize_model(m, label):
    params = m.params
    conf = m.conf_int()
    pvals = m.pvalues
    # odds ratios
    or_vals = np.exp(params)
    or_conf = np.exp(conf)
    summary = {
        'label': label,
        'n': int(m.nobs),
        'params': params.to_dict(),
        'pvalues': pvals.to_dict(),
        'odds_ratios': or_vals.to_dict(),
        'odds_ratio_ci': {k: [float(or_conf.loc[k, 0]), float(or_conf.loc[k, 1])] for k in or_conf.index},
        'pseudo_r2': float(m.prsquared),
        'llr_pvalue': float(m.llr_pvalue),
    }
    return summary

res = {
    'model_size_diff': summarize_model(model, 'size_diff'),
    'model_size_log_ratio': summarize_model(model2, 'size_log_ratio'),
    'descriptives': {
        'win_rate': float(df[outcome].mean()),
        'n': int(len(df)),
        'size_diff_mean': float(df['size_diff'].mean()),
        'loc_adv_mean': float(df['loc_adv'].mean()),
    }
}

print(json.dumps(res, indent=2))
