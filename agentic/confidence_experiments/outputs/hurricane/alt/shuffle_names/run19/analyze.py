import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map shuffled columns to meaningful names based on metadata/value ranges
# (See AGENTS.md: use info.json but columns are shuffled in this dataset.)
df = df.rename(columns={
    'wind': 'year',                # 1950-2012
    'alldeaths': 'name',           # string hurricane name
    'category': 'femininity',      # 1-11 scale
    'ndam15': 'min_pressure',      # 909-1002 mb
    'masfem_mturk': 'female_name', # binary gender indicator
    'gender_mf': 'saffir_cat',     # 1-5 category
    'name': 'deaths',              # total deaths
    'elapsedyrs': 'damage_2013',   # normalized damage
    'masfem': 'elapsed_years',     # years since hurricane
    'min': 'source',               # data source
    'ind': 'femininity_mturk',     # MTurk rating 1-11
    'year': 'wind_speed',          # max wind speed
    'source': 'damage_2015'        # normalized damage
})

# Outcome
# Use log(1+deaths) to reduce skew

df['log_deaths'] = np.log1p(df['deaths'])

# Build models with severity controls
controls = ['saffir_cat', 'wind_speed', 'min_pressure', 'year']

results = {}

# Model with continuous femininity rating
cols = ['femininity'] + controls
model_df = df[cols + ['log_deaths']].dropna()
X = sm.add_constant(model_df[cols])
y = model_df['log_deaths']
model = sm.OLS(y, X).fit(cov_type='HC3')
results['femininity_cont'] = {
    'n': int(model.nobs),
    'coef': float(model.params['femininity']),
    'pval': float(model.pvalues['femininity']),
    'stderr': float(model.bse['femininity']),
    'r2': float(model.rsquared)
}

# Model with female vs male indicator
cols = ['female_name'] + controls
model_df = df[cols + ['log_deaths']].dropna()
X = sm.add_constant(model_df[cols])
y = model_df['log_deaths']
model = sm.OLS(y, X).fit(cov_type='HC3')
results['female_indicator'] = {
    'n': int(model.nobs),
    'coef': float(model.params['female_name']),
    'pval': float(model.pvalues['female_name']),
    'stderr': float(model.bse['female_name']),
    'r2': float(model.rsquared)
}

# Model with MTurk femininity rating
cols = ['femininity_mturk'] + controls
model_df = df[cols + ['log_deaths']].dropna()
X = sm.add_constant(model_df[cols])
y = model_df['log_deaths']
model = sm.OLS(y, X).fit(cov_type='HC3')
results['femininity_mturk'] = {
    'n': int(model.nobs),
    'coef': float(model.params['femininity_mturk']),
    'pval': float(model.pvalues['femininity_mturk']),
    'stderr': float(model.bse['femininity_mturk']),
    'r2': float(model.rsquared)
}

# Simple correlations (Spearman) for robustness
corrs = {}
for var in ['femininity', 'female_name', 'femininity_mturk']:
    sub = df[[var, 'deaths']].dropna()
    corr = sub[var].corr(sub['deaths'], method='spearman')
    corrs[var] = float(corr)

print('Model results:')
for k, v in results.items():
    print(k, v)

print('Spearman correlations with deaths:', corrs)
