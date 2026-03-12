import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
DF = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = ['masfem','masfem_mturk','gender_mf','alldeaths','wind','min','category','ndam','ndam15','elapsedyrs']
for c in num_cols:
    if c in DF.columns:
        DF[c] = pd.to_numeric(DF[c], errors='coerce')

# Create log death outcome
DF['log_alldeaths'] = np.log1p(DF['alldeaths'])

# Simple correlations
corr_masfem = DF[['masfem','alldeaths','log_alldeaths']].corr()

# OLS models
# Base: log deaths ~ masfem
m1 = smf.ols('log_alldeaths ~ masfem', data=DF).fit()

# Controls for severity: wind + min + category
m2 = smf.ols('log_alldeaths ~ masfem + wind + min + category', data=DF).fit()

# Alternative: use damage (ndam15) as severity/economic impact proxy
m3 = smf.ols('log_alldeaths ~ masfem + wind + min + category + ndam15', data=DF).fit()

# Gender binary
m4 = smf.ols('log_alldeaths ~ gender_mf + wind + min + category', data=DF).fit()

# Poisson and NegBin on counts
# Add small constant to avoid issues with zeros for log link? Poisson handles zeros.
poisson = smf.glm('alldeaths ~ masfem + wind + min + category', data=DF, family=sm.families.Poisson()).fit()

# Negative binomial using GLM with NB family
nb = smf.glm('alldeaths ~ masfem + wind + min + category', data=DF, family=sm.families.NegativeBinomial()).fit()

# Collect key results
results = {
    'n_rows': int(DF.shape[0]),
    'corr_masfem_alldeaths': float(corr_masfem.loc['masfem','alldeaths']),
    'corr_masfem_log_alldeaths': float(corr_masfem.loc['masfem','log_alldeaths']),
    'm1_coef': float(m1.params.get('masfem', np.nan)),
    'm1_p': float(m1.pvalues.get('masfem', np.nan)),
    'm2_coef': float(m2.params.get('masfem', np.nan)),
    'm2_p': float(m2.pvalues.get('masfem', np.nan)),
    'm3_coef': float(m3.params.get('masfem', np.nan)),
    'm3_p': float(m3.pvalues.get('masfem', np.nan)),
    'm4_coef': float(m4.params.get('gender_mf', np.nan)),
    'm4_p': float(m4.pvalues.get('gender_mf', np.nan)),
    'poisson_coef': float(poisson.params.get('masfem', np.nan)),
    'poisson_p': float(poisson.pvalues.get('masfem', np.nan)),
    'nb_coef': float(nb.params.get('masfem', np.nan)),
    'nb_p': float(nb.pvalues.get('masfem', np.nan)),
}

print('RESULTS_START')
for k, v in results.items():
    print(f'{k}: {v}')
print('RESULTS_END')

# Save a compact stats summary to review if needed
with open('analysis_summary.txt', 'w') as f:
    f.write(m2.summary().as_text())
    f.write('\n\n')
    f.write(m3.summary().as_text())
    f.write('\n\n')
    f.write(poisson.summary().as_text())
    f.write('\n\n')
    f.write(nb.summary().as_text())
