import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'hurricane.csv'

df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = ['masfem','masfem_mturk','gender_mf','category','alldeaths','ndam','ndam15','wind','min','year','elapsedyrs']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Create log1p deaths
# Some analyses use log( deaths + 1 )
df['log_deaths'] = np.log1p(df['alldeaths'])

# Center wind and masfem for interaction stability
if 'wind' in df.columns:
    df['wind_c'] = df['wind'] - df['wind'].mean()
if 'masfem' in df.columns:
    df['masfem_c'] = df['masfem'] - df['masfem'].mean()

# Basic correlations
corrs = df[['masfem','alldeaths','log_deaths','wind','min','category','ndam15']].corr()

# Define several models
models = {}

# Model 1: log_deaths ~ masfem
models['m1'] = smf.ols('log_deaths ~ masfem', data=df).fit()

# Model 2: log_deaths ~ masfem + wind
models['m2'] = smf.ols('log_deaths ~ masfem + wind', data=df).fit()

# Model 3: log_deaths ~ masfem + category
models['m3'] = smf.ols('log_deaths ~ masfem + category', data=df).fit()

# Model 4: log_deaths ~ masfem + wind + year (to adjust for changes in warning systems)
models['m4'] = smf.ols('log_deaths ~ masfem + wind + year', data=df).fit()

# Model 5: log_deaths ~ masfem + wind + min (pressure)
# May be collinear; still check
models['m5'] = smf.ols('log_deaths ~ masfem + wind + min', data=df).fit()

# Interaction model: log_deaths ~ masfem * wind
models['m6'] = smf.ols('log_deaths ~ masfem_c * wind_c', data=df).fit()

# Alternative: use gender_mf
models['m7'] = smf.ols('log_deaths ~ gender_mf + wind', data=df).fit()

# Print summaries
print('N:', len(df))
print('\nCorrelations (subset):')
print(corrs)

for name, m in models.items():
    print(f"\n{name} summary (key coefficients):")
    # extract coefficients of interest
    params = m.params
    pvals = m.pvalues
    for term in ['masfem','masfem_c','gender_mf','wind','wind_c','category','year','min','masfem_c:wind_c']:
        if term in params.index:
            print(f"  {term}: coef={params[term]:.4f}, p={pvals[term]:.4g}")
    print(f"  R2={m.rsquared:.4f}")

# Also check Poisson regression on counts
# Sometimes deaths are count data; use log link
# We'll include wind and masfem
try:
    poisson = smf.glm('alldeaths ~ masfem + wind', data=df, family=sm.families.Poisson()).fit()
    print('\nPoisson GLM (alldeaths ~ masfem + wind)')
    for term in ['masfem','wind']:
        if term in poisson.params.index:
            print(f"  {term}: coef={poisson.params[term]:.4f}, p={poisson.pvalues[term]:.4g}")
    print(f"  AIC={poisson.aic:.2f}")
except Exception as e:
    print('Poisson model failed:', e)

# Another: negative binomial if available
try:
    nb = smf.glm('alldeaths ~ masfem + wind', data=df, family=sm.families.NegativeBinomial()).fit()
    print('\nNegBin GLM (alldeaths ~ masfem + wind)')
    for term in ['masfem','wind']:
        if term in nb.params.index:
            print(f"  {term}: coef={nb.params[term]:.4f}, p={nb.pvalues[term]:.4g}")
    print(f"  AIC={nb.aic:.2f}")
except Exception as e:
    print('NegBin model failed:', e)

