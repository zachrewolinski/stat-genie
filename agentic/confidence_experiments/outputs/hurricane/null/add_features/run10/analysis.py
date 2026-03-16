import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Keep relevant columns and drop missing
cols = ['alldeaths','masfem','gender_mf','wind','min','category','ndam','ndam15']

# Ensure numeric
for c in cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Base outcome

df['log_deaths'] = np.log1p(df['alldeaths'])

# Basic model: log deaths vs masfem
model1 = smf.ols('log_deaths ~ masfem', data=df).fit(cov_type='HC3')

# Add controls for storm severity
model2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit(cov_type='HC3')

# Interaction with severity (wind)
model3 = smf.ols('log_deaths ~ masfem * wind + min + category', data=df).fit(cov_type='HC3')

# Binary gender
model4 = smf.ols('log_deaths ~ gender_mf + wind + min + category', data=df).fit(cov_type='HC3')

model5 = smf.ols('log_deaths ~ gender_mf * wind + min + category', data=df).fit(cov_type='HC3')

# Poisson regression for count outcome
# Add 1 to avoid zero issues? Poisson can handle zeros.
try:
    pois1 = smf.glm('alldeaths ~ masfem + wind + min + category', data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
    pois2 = smf.glm('alldeaths ~ masfem * wind + min + category', data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
except Exception as e:
    pois1 = None
    pois2 = None

# Summaries

def summarize(model, name):
    if model is None:
        return None
    params = model.params
    b_masfem = params.get('masfem', np.nan)
    p_masfem = model.pvalues.get('masfem', np.nan)
    b_int = params.get('masfem:wind', np.nan)
    p_int = model.pvalues.get('masfem:wind', np.nan)
    return {
        'name': name,
        'n': int(model.nobs),
        'b_masfem': float(b_masfem),
        'p_masfem': float(p_masfem),
        'b_int': float(b_int),
        'p_int': float(p_int),
        'r2': float(getattr(model, 'rsquared', np.nan)),
        'aic': float(model.aic)
    }

summaries = [
    summarize(model1, 'OLS log deaths ~ masfem'),
    summarize(model2, 'OLS log deaths ~ masfem + controls'),
    summarize(model3, 'OLS log deaths ~ masfem*wind + controls'),
    summarize(model4, 'OLS log deaths ~ gender + controls'),
    summarize(model5, 'OLS log deaths ~ gender*wind + controls'),
    summarize(pois1, 'Poisson deaths ~ masfem + controls') if pois1 is not None else None,
    summarize(pois2, 'Poisson deaths ~ masfem*wind + controls') if pois2 is not None else None,
]

summaries = [s for s in summaries if s is not None]

print('Rows:', len(df))
print('Deaths summary:', df['alldeaths'].describe())
print('\nModel summaries:')
for s in summaries:
    print(s)

# Also compute correlation between masfem and deaths
corr = df[['masfem','alldeaths']].corr().iloc[0,1]
print('\nCorrelation masfem vs deaths:', corr)

