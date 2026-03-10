import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

print('rows', len(df))
print('columns', list(df.columns))

# Select relevant columns
cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'ndam', 'ndam15', 'elapsedyrs']
missing = [c for c in cols if c not in df.columns]
print('missing cols', missing)

analysis_df = df[cols].copy()
print('missing counts:\n', analysis_df.isna().sum())

# Transformations
analysis_df['log_deaths'] = np.log1p(analysis_df['alldeaths'])
analysis_df['log_ndam'] = np.log1p(analysis_df['ndam'])
analysis_df['log_ndam15'] = np.log1p(analysis_df['ndam15'])

print('corr masfem vs log_deaths', analysis_df[['masfem','log_deaths']].corr().iloc[0,1])
print('corr gender_mf vs log_deaths', analysis_df[['gender_mf','log_deaths']].corr().iloc[0,1])

# Helper to fit OLS with dropna

def fit_ols(y, X_cols, data):
    sub = data[[y] + X_cols].dropna().copy()
    X = sm.add_constant(sub[X_cols])
    model = sm.OLS(sub[y], X).fit()
    return model, len(sub)

models = {}

# Model 1: log_deaths ~ masfem
models['m1'], n1 = fit_ols('log_deaths', ['masfem'], analysis_df)

# Model 2: severity controls
models['m2'], n2 = fit_ols('log_deaths', ['masfem','wind','min','category'], analysis_df)

# Model 3: add damages + elapsed years
models['m3'], n3 = fit_ols('log_deaths', ['masfem','wind','min','category','log_ndam15','elapsedyrs'], analysis_df)

# Model 4: gender_mf instead of masfem
models['m4'], n4 = fit_ols('log_deaths', ['gender_mf','wind','min','category'], analysis_df)

n_map = {'m1': n1, 'm2': n2, 'm3': n3, 'm4': n4}

for name, m in models.items():
    print('\n', name, 'n', n_map[name])
    print('coef')
    print(m.params)
    print('pvalues')
    print(m.pvalues)
    print('R2', m.rsquared)

# Robust SEs for m2 and m3
for name in ['m2','m3']:
    m = models[name]
    rob = m.get_robustcov_results(cov_type='HC3')
    print('\n', name, 'robust')
    print('coef')
    print(rob.params)
    print('pvalues')
    print(rob.pvalues)

# Poisson regression for counts
sub_pois = analysis_df[['alldeaths','masfem','wind','min','category']].dropna().copy()
poisson = smf.glm('alldeaths ~ masfem + wind + min + category', data=sub_pois, family=sm.families.Poisson()).fit()
print('\npoisson n', len(sub_pois))
print('coef')
print(poisson.params)
print('pvalues')
print(poisson.pvalues)

# Negative binomial
try:
    nb = smf.glm('alldeaths ~ masfem + wind + min + category', data=sub_pois, family=sm.families.NegativeBinomial()).fit()
    print('\nneg bin')
    print('coef')
    print(nb.params)
    print('pvalues')
    print(nb.pvalues)
except Exception as e:
    print('nb error', e)

