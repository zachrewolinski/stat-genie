import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('crofoot.csv')

# Create predictors
# Relative group size: difference and ratio

df['size_diff'] = df['n_focal'] - df['n_other']
df['size_ratio'] = df['n_focal'] / df['n_other']

# Location advantage: positive if contest closer to focal center than other center

df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Standardize continuous predictors for comparability
for col in ['size_diff', 'loc_adv']:
    df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression with size_diff and loc_adv
model = smf.logit('win ~ z_size_diff + z_loc_adv', data=df).fit(disp=False)

# Also test size_ratio and loc_adv

df['z_size_ratio'] = (df['size_ratio'] - df['size_ratio'].mean()) / df['size_ratio'].std(ddof=0)
model_ratio = smf.logit('win ~ z_size_ratio + z_loc_adv', data=df).fit(disp=False)

# Simple models for each predictor
model_size_only = smf.logit('win ~ z_size_diff', data=df).fit(disp=False)
model_loc_only = smf.logit('win ~ z_loc_adv', data=df).fit(disp=False)

# Pseudo R2 (McFadden)

def mcfadden(m):
    return 1 - m.llf / m.llnull

results = {
    'n': len(df),
    'model': model.params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_or': np.exp(model.params).to_dict(),
    'model_conf': model.conf_int().rename(columns={0: 'low', 1: 'high'}).to_dict(),
    'model_mcfadden': mcfadden(model),
    'model_ratio': model_ratio.params.to_dict(),
    'model_ratio_pvalues': model_ratio.pvalues.to_dict(),
    'model_ratio_or': np.exp(model_ratio.params).to_dict(),
    'model_ratio_mcfadden': mcfadden(model_ratio),
    'model_size_only_pvalues': model_size_only.pvalues.to_dict(),
    'model_loc_only_pvalues': model_loc_only.pvalues.to_dict(),
}

# Predictive effect: marginal effect for 1 SD change
me = model.get_margeff(at='mean').summary_frame()
results['marginal_effects'] = me.to_dict()

# Crosstab for intuitive check: win rate by size_diff sign and loc_adv sign
df['size_diff_sign'] = np.where(df['size_diff'] >= 0, 'focal>=other', 'focal<other')
df['loc_adv_sign'] = np.where(df['loc_adv'] >= 0, 'closer_to_focal', 'closer_to_other')

ctab = df.groupby(['size_diff_sign', 'loc_adv_sign'])['win'].agg(['mean', 'count'])
results['ctab'] = ctab.reset_index().to_dict(orient='records')

print(results)
