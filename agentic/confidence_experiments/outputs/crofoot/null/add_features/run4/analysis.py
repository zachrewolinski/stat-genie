import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: keep relevant columns and drop missing
cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = df[cols].dropna().copy()

# Derived variables
# Relative group size: focal minus other
# Also consider ratio for robustness

df['rel_size'] = df['n_focal'] - df['n_other']
# Relative location: focal closer to its home range center than other (smaller distance)
# 1 if focal closer to its center than other group, 0 otherwise

df['focal_closer'] = (df['dist_focal'] < df['dist_other']).astype(int)

# Also distance difference (positive = focal farther from its center)

df['dist_diff'] = df['dist_focal'] - df['dist_other']

# Logistic regression with rel_size and focal_closer
model1 = smf.logit('win ~ rel_size + focal_closer', data=df).fit(disp=False)

# Logistic regression with rel_size and distance difference (continuous)
model2 = smf.logit('win ~ rel_size + dist_diff', data=df).fit(disp=False)

# For effect size interpretation, compute odds ratios and p-values

def summarize(model):
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    odds = np.exp(params)
    conf_odds = np.exp(conf)
    summary = pd.DataFrame({
        'coef': params,
        'odds_ratio': odds,
        'pvalue': pvals,
        'conf_low_or': conf_odds[0],
        'conf_high_or': conf_odds[1],
    })
    return summary

summary1 = summarize(model1)
summary2 = summarize(model2)

# basic counts
n = len(df)
win_rate = df['win'].mean()

print('N', n)
print('Win rate', win_rate)
print('\nModel1: win ~ rel_size + focal_closer')
print(summary1)
print('\nModel2: win ~ rel_size + dist_diff')
print(summary2)

# also check simple bivariate relationships
# logistic with rel_size only and focal_closer only
model_rel = smf.logit('win ~ rel_size', data=df).fit(disp=False)
model_loc = smf.logit('win ~ focal_closer', data=df).fit(disp=False)

print('\nModel rel_size only')
print(summarize(model_rel))
print('\nModel focal_closer only')
print(summarize(model_loc))

# also check with standardized dist_diff

df['dist_diff_z'] = (df['dist_diff'] - df['dist_diff'].mean())/df['dist_diff'].std()
model2z = smf.logit('win ~ rel_size + dist_diff_z', data=df).fit(disp=False)
print('\nModel2z: win ~ rel_size + dist_diff_z')
print(summarize(model2z))

# Save key stats for later use in manual reasoning
stats = {
    'n': n,
    'win_rate': float(win_rate),
    'model1_params': summary1.to_dict(orient='index'),
    'model2_params': summary2.to_dict(orient='index'),
    'model_rel_params': summarize(model_rel).to_dict(orient='index'),
    'model_loc_params': summarize(model_loc).to_dict(orient='index'),
    'model2z_params': summarize(model2z).to_dict(orient='index'),
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(stats, f, indent=2)
