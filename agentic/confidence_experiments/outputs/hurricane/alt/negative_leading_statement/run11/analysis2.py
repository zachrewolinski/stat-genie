import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Poisson model
poisson = smf.glm('alldeaths ~ masfem + wind + min + category', data=_df,
                  family=sm.families.Poisson()).fit()

dispersion = poisson.deviance / poisson.df_resid

# Negative binomial model (NB2)
# statsmodels NegativeBinomial can be fit via GLM with NegativeBinomial(alpha=...)
# We'll let statsmodels estimate alpha using discrete model as well

# GLM NB with alpha estimated by method of moments from Poisson? use discrete NB to estimate alpha
nb2 = smf.glm('alldeaths ~ masfem + wind + min + category', data=_df,
              family=sm.families.NegativeBinomial()).fit()

# Discrete NegativeBinomial (NB2) estimation
nb_discrete = smf.negativebinomial('alldeaths ~ masfem + wind + min + category', data=_df).fit(disp=0)

out = {
    'poisson': {
        'coef': float(poisson.params['masfem']),
        'se': float(poisson.bse['masfem']),
        'pvalue': float(poisson.pvalues['masfem']),
        'dispersion': float(dispersion),
    },
    'nb_glm': {
        'coef': float(nb2.params['masfem']),
        'se': float(nb2.bse['masfem']),
        'pvalue': float(nb2.pvalues['masfem']),
        'alpha': float(nb2.scale),
    },
    'nb_discrete': {
        'coef': float(nb_discrete.params['masfem']),
        'se': float(nb_discrete.bse['masfem']),
        'pvalue': float(nb_discrete.pvalues['masfem']),
        'alpha': float(nb_discrete.params['alpha']),
    }
}

import json
with open('analysis_count_models.json','w') as f:
    json.dump(out, f, indent=2)

print(out)
