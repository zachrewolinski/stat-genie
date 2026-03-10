import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('crofoot.csv')

# Keep relevant columns and drop rows with missing values
cols = ['win','n_focal','n_other','dist_focal','dist_other']

# Ensure numeric
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Create relative size and location variables
# size_diff: focal group size minus other group size
# size_ratio: focal / other (avoid division by zero)
# dist_diff: other distance minus focal distance (positive => focal closer to its center)

# Use size_diff and dist_diff for modeling

df['size_diff'] = df['n_focal'] - df['n_other']
# If n_other == 0 (shouldn't), avoid division

df['size_ratio'] = df['n_focal'] / df['n_other']

df['dist_diff'] = df['dist_other'] - df['dist_focal']

# Drop missing
model_df = df[['win','size_diff','size_ratio','dist_diff']].dropna()

# Logistic regression with size_diff and dist_diff
# Add standardized versions to ease interpretation
model_df['size_diff_z'] = (model_df['size_diff'] - model_df['size_diff'].mean())/model_df['size_diff'].std(ddof=0)
model_df['dist_diff_z'] = (model_df['dist_diff'] - model_df['dist_diff'].mean())/model_df['dist_diff'].std(ddof=0)

# Fit model
logit = smf.logit('win ~ size_diff_z + dist_diff_z', data=model_df).fit(disp=False)

# Also check size_ratio instead of size_diff to confirm robustness
model_df['size_ratio_z'] = (model_df['size_ratio'] - model_df['size_ratio'].mean())/model_df['size_ratio'].std(ddof=0)
logit_ratio = smf.logit('win ~ size_ratio_z + dist_diff_z', data=model_df).fit(disp=False)

# Simple bivariate checks
logit_size_only = smf.logit('win ~ size_diff_z', data=model_df).fit(disp=False)
logit_dist_only = smf.logit('win ~ dist_diff_z', data=model_df).fit(disp=False)

# Collect results

def summarize_model(m):
    params = m.params
    conf = m.conf_int()
    pvalues = m.pvalues
    return pd.DataFrame({
        'coef': params,
        'p': pvalues,
        'ci_low': conf[0],
        'ci_high': conf[1]
    })

summary_main = summarize_model(logit)
summary_ratio = summarize_model(logit_ratio)
summary_size_only = summarize_model(logit_size_only)
summary_dist_only = summarize_model(logit_dist_only)

# Compute pseudo R2 (McFadden)

pseudo_r2_main = logit.prsquared
pseudo_r2_ratio = logit_ratio.prsquared

# Save key outputs

print('N:', len(model_df))
print('\nMain model (size_diff_z + dist_diff_z):')
print(summary_main)
print('Pseudo R2:', pseudo_r2_main)

print('\nRatio model (size_ratio_z + dist_diff_z):')
print(summary_ratio)
print('Pseudo R2:', pseudo_r2_ratio)

print('\nSize only:')
print(summary_size_only)
print('\nDist only:')
print(summary_dist_only)
