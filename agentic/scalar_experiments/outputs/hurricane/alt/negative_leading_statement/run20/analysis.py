import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path('hurricane.csv')

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Use alldeaths as outcome; add log1p to handle zeros
# Use masfem as primary femininity measure; also check masfem_mturk and gender_mf

df['log_deaths'] = np.log1p(df['alldeaths'])

# Some variables for severity
# min pressure (lower => stronger), wind, category

# Prepare a few models
results = {}

# Model 1: bivariate regression log_deaths ~ masfem
model1 = smf.ols('log_deaths ~ masfem', data=df).fit()
results['model1'] = model1

# Model 2: add severity controls
model2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit()
results['model2'] = model2

# Model 3: add year (or elapsedyrs) to account for time trend
model3 = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=df).fit()
results['model3'] = model3

# Model 4: alternate femininity measure (masfem_mturk)
model4 = smf.ols('log_deaths ~ masfem_mturk + wind + min + category', data=df).fit()
results['model4'] = model4

# Model 5: gender_mf binary
model5 = smf.ols('log_deaths ~ gender_mf + wind + min + category', data=df).fit()
results['model5'] = model5

# Count models for deaths (Poisson and Negative Binomial)
model6 = smf.glm('alldeaths ~ masfem + wind + min + category', data=df,
                 family=sm.families.Poisson()).fit()
results['model6_poisson'] = model6

model7 = smf.glm('alldeaths ~ masfem + wind + min + category', data=df,
                 family=sm.families.NegativeBinomial()).fit()
results['model7_nb_glm'] = model7

# Discrete negative binomial with estimated dispersion
model7b = smf.negativebinomial('alldeaths ~ masfem + wind + min + category', data=df).fit(disp=0)
results['model7b_nb_discrete'] = model7b

# Interaction with severity (wind)
model8 = smf.ols('log_deaths ~ masfem * wind + min + category', data=df).fit()
results['model8_interaction'] = model8

# Count models with year control
model9 = smf.glm('alldeaths ~ masfem + wind + min + category + year', data=df,
                 family=sm.families.Poisson()).fit()
results['model9_poisson_year'] = model9

model10 = smf.negativebinomial('alldeaths ~ masfem + wind + min + category + year', data=df).fit(disp=0)
results['model10_nb_year'] = model10

# Robust linear model on log deaths
model11 = smf.rlm('log_deaths ~ masfem + wind + min + category', data=df).fit()
results['model11_rlm'] = model11

# Also check correlations
corrs = df[['masfem','masfem_mturk','gender_mf','alldeaths','log_deaths','wind','min','category']].corr(numeric_only=True)

# Summaries for key coefficients
summary = {}
for name, model in results.items():
    keys = [k for k in ['masfem', 'masfem_mturk', 'gender_mf', 'masfem:wind'] if k in model.params.index]
    if not keys:
        continue
    summary[name] = {
        'n': int(model.nobs),
        'r2': float(getattr(model, 'rsquared', float('nan'))),
        'terms': {}
    }
    for key in keys:
        summary[name]['terms'][key] = {
            'coef': float(model.params[key]),
            'pvalue': float(model.pvalues[key]),
            'stderr': float(model.bse[key]),
        }

# Save results to json for later reading
out = {
    'corrs': corrs.loc[['masfem','masfem_mturk','gender_mf'], ['alldeaths','log_deaths','wind','min','category']].to_dict(),
    'summary': summary
}

# Spearman correlations
spearman = {}
for var in ['masfem', 'masfem_mturk', 'gender_mf']:
    rho, p = stats.spearmanr(df[var], df['alldeaths'])
    spearman[var] = {'rho': float(rho), 'pvalue': float(p)}
out['spearman'] = spearman

Path('analysis_results.json').write_text(json.dumps(out, indent=2))

print(json.dumps(out, indent=2))
