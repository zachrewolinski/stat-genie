import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('hurricane.csv')

# Basic numeric conversions
numeric_cols = [
    'masfem','masfem_mturk','gender_mf','category','alldeaths','ndam','ndam15',
    'wind','min','year','elapsedyrs'
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Create log outcomes
for col in ['alldeaths','ndam','ndam15']:
    if col in df.columns:
        df[f'log1p_{col}'] = np.log1p(df[col])

# Helper for OLS with robust SE

def run_ols(y, x_cols, data):
    X = data[x_cols].copy()
    X = sm.add_constant(X, has_constant='add')
    yv = data[y]
    model = sm.OLS(yv, X, missing='drop')
    res = model.fit(cov_type='HC3')
    return res

results = []

# Model specs
specs = [
    # Deaths
    ('log1p_alldeaths', ['masfem','wind','min','year']),
    ('log1p_alldeaths', ['masfem','wind','category','year']),
    ('log1p_alldeaths', ['gender_mf','wind','min','year']),
    ('log1p_alldeaths', ['gender_mf','wind','category','year']),
    # Damages (2015 adjusted)
    ('log1p_ndam15', ['masfem','wind','min','year']),
    ('log1p_ndam15', ['masfem','wind','category','year']),
    ('log1p_ndam15', ['gender_mf','wind','min','year']),
    ('log1p_ndam15', ['gender_mf','wind','category','year']),
]

for y, x in specs:
    if y not in df.columns:
        continue
    res = run_ols(y, x, df)
    results.append((y, x, res))

# Output summary stats
print('N rows:', len(df))
print('Missing in key vars:', df[['masfem','gender_mf','alldeaths','ndam15','wind','min','category','year']].isna().sum().to_dict())

for y, x, res in results:
    var = 'masfem' if 'masfem' in x else 'gender_mf'
    coef = res.params.get(var, np.nan)
    se = res.bse.get(var, np.nan)
    pval = res.pvalues.get(var, np.nan)
    r2 = res.rsquared
    print('\nModel:', y, '~', ' + '.join(x))
    print('coef', var, ':', coef, 'SE:', se, 'p:', pval, 'R2:', r2)

# Also simple correlations
corr_cols = ['masfem','gender_mf','alldeaths','ndam15','wind','min','category']
sub = df[corr_cols].dropna()
print('\nCorrelations (pearson):')
print(sub.corr())
