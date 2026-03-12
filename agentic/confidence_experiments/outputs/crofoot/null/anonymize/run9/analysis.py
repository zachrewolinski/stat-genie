import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data

df = pd.read_csv('crofoot.csv')

# Outcome: focal win
# Predictors
# Relative group size: difference in total individuals and log ratio
# Contest location: distance from each group's home-range center and relative location index
# Location advantage: other group's distance minus focal group's distance

df = df.assign(
    size_diff=df['feature7'] - df['feature8'],
    size_ratio=df['feature7'] / df['feature8'],
    loc_adv=df['feature6'] - df['feature5'],
)
df['size_log_ratio'] = np.log(df['size_ratio'])
df['loc_ratio'] = df['feature5'] / (df['feature5'] + df['feature6'])

# Standardize predictors for interpretability
for col in ['size_diff', 'size_log_ratio', 'loc_adv', 'loc_ratio', 'feature5', 'feature6']:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Fit logistic regression with both predictors
X = sm.add_constant(df[['size_diff_z', 'loc_adv_z']])
y = df['feature4']
model = sm.Logit(y, X).fit(disp=False)

# Single-predictor models
X_size = sm.add_constant(df[['size_diff_z']])
model_size = sm.Logit(y, X_size).fit(disp=False)

X_size_log = sm.add_constant(df[['size_log_ratio_z']])
model_size_log = sm.Logit(y, X_size_log).fit(disp=False)

X_loc_adv = sm.add_constant(df[['loc_adv_z']])
model_loc_adv = sm.Logit(y, X_loc_adv).fit(disp=False)

X_loc_ratio = sm.add_constant(df[['loc_ratio_z']])
model_loc_ratio = sm.Logit(y, X_loc_ratio).fit(disp=False)

# Separate focal/other distance model
X_loc_both = sm.add_constant(df[['feature5_z', 'feature6_z']])
model_loc_both = sm.Logit(y, X_loc_both).fit(disp=False)

# Compute effect on predicted probability for +/-1 SD (holding other at 0)

def pred_prob(model, size_z=0.0, loc_z=0.0):
    params = model.params
    # assumes const, size_diff_z, loc_adv_z (if present)
    x = [1.0]
    if 'size_diff_z' in params.index:
        x.append(size_z)
    if 'loc_adv_z' in params.index:
        x.append(loc_z)
    x = np.array(x)
    lin = np.dot(x, params.values)
    return 1.0 / (1.0 + np.exp(-lin))

# For joint model
p_low_size = pred_prob(model, size_z=-1, loc_z=0)
p_high_size = pred_prob(model, size_z=1, loc_z=0)
p_low_loc = pred_prob(model, size_z=0, loc_z=-1)
p_high_loc = pred_prob(model, size_z=0, loc_z=1)

summary = {
    'n': int(len(df)),
    'outcome_mean': float(y.mean()),
    'size_diff': {
        'mean': float(df['size_diff'].mean()),
        'std': float(df['size_diff'].std(ddof=0)),
    },
    'loc_adv': {
        'mean': float(df['loc_adv'].mean()),
        'std': float(df['loc_adv'].std(ddof=0)),
    },
    'joint_model': {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'llf': float(model.llf),
        'pseudo_r2': float(model.prsquared),
        'pred_prob_low_size': float(p_low_size),
        'pred_prob_high_size': float(p_high_size),
        'pred_prob_low_loc': float(p_low_loc),
        'pred_prob_high_loc': float(p_high_loc),
    },
    'size_model': {
        'params': model_size.params.to_dict(),
        'pvalues': model_size.pvalues.to_dict(),
        'llf': float(model_size.llf),
        'pseudo_r2': float(model_size.prsquared),
    },
    'size_log_model': {
        'params': model_size_log.params.to_dict(),
        'pvalues': model_size_log.pvalues.to_dict(),
        'llf': float(model_size_log.llf),
        'pseudo_r2': float(model_size_log.prsquared),
    },
    'loc_adv_model': {
        'params': model_loc_adv.params.to_dict(),
        'pvalues': model_loc_adv.pvalues.to_dict(),
        'llf': float(model_loc_adv.llf),
        'pseudo_r2': float(model_loc_adv.prsquared),
    },
    'loc_ratio_model': {
        'params': model_loc_ratio.params.to_dict(),
        'pvalues': model_loc_ratio.pvalues.to_dict(),
        'llf': float(model_loc_ratio.llf),
        'pseudo_r2': float(model_loc_ratio.prsquared),
    },
    'loc_both_model': {
        'params': model_loc_both.params.to_dict(),
        'pvalues': model_loc_both.pvalues.to_dict(),
        'llf': float(model_loc_both.llf),
        'pseudo_r2': float(model_loc_both.prsquared),
    },
}

print(json.dumps(summary, indent=2))
