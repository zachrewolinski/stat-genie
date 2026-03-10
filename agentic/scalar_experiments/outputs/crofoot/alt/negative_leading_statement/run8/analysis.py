import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DF = pd.read_csv('crofoot.csv')

# Derived variables
DF['rel_size'] = DF['n_focal'] - DF['n_other']
DF['loc_adv'] = DF['dist_other'] - DF['dist_focal']  # positive = contest closer to focal home range

# Standardize predictors for interpretability
DF['rel_size_z'] = (DF['rel_size'] - DF['rel_size'].mean()) / DF['rel_size'].std(ddof=0)
DF['loc_adv_z'] = (DF['loc_adv'] - DF['loc_adv'].mean()) / DF['loc_adv'].std(ddof=0)

# Logistic regression: win ~ rel_size + loc_adv
X = DF[['rel_size_z', 'loc_adv_z']]
X = sm.add_constant(X)
model = sm.GLM(DF['win'], X, family=sm.families.Binomial())
res = model.fit()

# Single predictor models
X_size = sm.add_constant(DF[['rel_size_z']])
res_size = sm.GLM(DF['win'], X_size, family=sm.families.Binomial()).fit()

X_loc = sm.add_constant(DF[['loc_adv_z']])
res_loc = sm.GLM(DF['win'], X_loc, family=sm.families.Binomial()).fit()

# Collect results
summary = {
    'n': int(len(DF)),
    'win_rate': float(DF['win'].mean()),
    'rel_size_mean': float(DF['rel_size'].mean()),
    'loc_adv_mean': float(DF['loc_adv'].mean()),
    'both_model': {
        'params': res.params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
        'aic': float(res.aic)
    },
    'size_only': {
        'params': res_size.params.to_dict(),
        'pvalues': res_size.pvalues.to_dict(),
        'aic': float(res_size.aic)
    },
    'loc_only': {
        'params': res_loc.params.to_dict(),
        'pvalues': res_loc.pvalues.to_dict(),
        'aic': float(res_loc.aic)
    }
}

# Save a concise report to stdout
print('N:', summary['n'])
print('Win rate:', round(summary['win_rate'], 3))
print('Rel size mean:', round(summary['rel_size_mean'], 3))
print('Loc advantage mean:', round(summary['loc_adv_mean'], 3))
print('\nBoth predictors (standardized):')
print('Params:', {k: round(v, 3) for k, v in summary['both_model']['params'].items()})
print('P-values:', {k: round(v, 4) for k, v in summary['both_model']['pvalues'].items()})
print('AIC:', round(summary['both_model']['aic'], 2))
print('\nRel size only:')
print('Params:', {k: round(v, 3) for k, v in summary['size_only']['params'].items()})
print('P-values:', {k: round(v, 4) for k, v in summary['size_only']['pvalues'].items()})
print('AIC:', round(summary['size_only']['aic'], 2))
print('\nLoc advantage only:')
print('Params:', {k: round(v, 3) for k, v in summary['loc_only']['params'].items()})
print('P-values:', {k: round(v, 4) for k, v in summary['loc_only']['pvalues'].items()})
print('AIC:', round(summary['loc_only']['aic'], 2))
