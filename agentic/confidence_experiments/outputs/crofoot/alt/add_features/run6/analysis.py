import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('crofoot.csv')

# Construct predictors
_df['size_diff'] = _df['n_focal'] - _df['n_other']  # relative group size (positive means focal larger)
_df['dist_diff'] = _df['dist_other'] - _df['dist_focal']  # positive means contest closer to focal

# Standardized versions for comparable effect sizes
for col in ['size_diff', 'dist_diff']:
    _df[col + '_z'] = (_df[col] - _df[col].mean()) / _df[col].std(ddof=0)

# Logistic regression (GLM binomial)
model = smf.glm('win ~ size_diff + dist_diff', data=_df, family=sm.families.Binomial()).fit()
model_z = smf.glm('win ~ size_diff_z + dist_diff_z', data=_df, family=sm.families.Binomial()).fit()

# Alternate location measure: proportion of distance to focal
_df['dist_prop_focal'] = _df['dist_focal'] / (_df['dist_focal'] + _df['dist_other'])
model_prop = smf.glm('win ~ size_diff + dist_prop_focal', data=_df, family=sm.families.Binomial()).fit()

# Basic summaries
summary = {
    'n': int(_df.shape[0]),
    'wins': int(_df['win'].sum()),
    'losses': int((1 - _df['win']).sum()),
}

# Collect key stats

def extract(model):
    params = model.params
    bse = model.bse
    p = model.pvalues
    odds = np.exp(params)
    return pd.DataFrame({
        'coef': params,
        'se': bse,
        'p': p,
        'odds_ratio': odds,
    })

print('SUMMARY', summary)
print('\nMODEL size_diff + dist_diff')
print(extract(model))
print('\nMODEL standardized')
print(extract(model_z))
print('\nMODEL size_diff + dist_prop_focal')
print(extract(model_prop))

# Model fit comparison
print('\nPseudo R2 (McFadden) model:', 1 - model.deviance / model.null_deviance)
print('Pseudo R2 (McFadden) model_prop:', 1 - model_prop.deviance / model_prop.null_deviance)
