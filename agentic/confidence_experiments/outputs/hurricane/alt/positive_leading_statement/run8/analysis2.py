import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

num_cols = ['masfem','masfem_mturk','alldeaths','wind','min','category','year','ndam','ndam15','elapsedyrs']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Negative binomial GLM
nb = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

# Also log1p OLS with additional control for ndam15? (damage) not good.

import json
res = {
    'nb': {
        'coef': float(nb.params.get('masfem', np.nan)),
        'se': float(nb.bse.get('masfem', np.nan)),
        'p': float(nb.pvalues.get('masfem', np.nan))
    },
    'alpha': float(nb.scale)
}

print(json.dumps(res, indent=2))
