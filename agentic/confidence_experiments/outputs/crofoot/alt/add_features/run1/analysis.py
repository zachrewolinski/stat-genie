import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Keep relevant rows with non-missing in key columns
cols = ['win','n_focal','n_other','dist_focal','dist_other']
df = _df[cols].dropna().copy()

# Derived variables
# Relative group size: focal - other
# Relative location: focal distance - other distance (negative means focal closer to its home range center)
df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_dist'] = df['dist_focal'] - df['dist_other']

# Also compute ratios for robustness
# Add small constant to avoid division by zero (not needed here but safe)
df['size_ratio'] = df['n_focal'] / df['n_other']

# Standardize for easier interpretation
for col in ['rel_size','rel_dist']:
    df[f'z_{col}'] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression: win ~ rel_size + rel_dist
model = smf.glm('win ~ z_rel_size + z_rel_dist', data=df, family=sm.families.Binomial())
res = model.fit()

# Univariate models
model_size = smf.glm('win ~ z_rel_size', data=df, family=sm.families.Binomial())
res_size = model_size.fit()

model_dist = smf.glm('win ~ z_rel_dist', data=df, family=sm.families.Binomial())
res_dist = model_dist.fit()

# Alternative model using ratio and rel_dist
model2 = smf.glm('win ~ size_ratio + z_rel_dist', data=df, family=sm.families.Binomial())
res2 = model2.fit()

# Null model for pseudo R2 (McFadden)
null_model = smf.glm('win ~ 1', data=df, family=sm.families.Binomial())
null_res = null_model.fit()

# Compute odds ratios and CI for main model
params = res.params
conf = res.conf_int()

odds = np.exp(params)
conf_odds = np.exp(conf)

# Collect results
out = {
    'n': int(df.shape[0]),
    'mean_win': float(df['win'].mean()),
    'model_summary': {
        'coef': params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
        'odds_ratio': odds.to_dict(),
        'odds_ci_low': conf_odds[0].to_dict(),
        'odds_ci_high': conf_odds[1].to_dict(),
        'llf': float(res.llf),
        'llf_null': float(null_res.llf),
        'mcfadden_r2': float(1 - res.llf/null_res.llf)
    },
    'size_only': {
        'coef': res_size.params.to_dict(),
        'pvalues': res_size.pvalues.to_dict(),
    },
    'dist_only': {
        'coef': res_dist.params.to_dict(),
        'pvalues': res_dist.pvalues.to_dict(),
    },
    'model2_summary': {
        'coef': res2.params.to_dict(),
        'pvalues': res2.pvalues.to_dict(),
    },
}

print(out)
