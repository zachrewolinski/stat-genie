import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Prepare variables
# Use log1p of deaths to handle skew and zeros

df['log_deaths'] = np.log1p(df['alldeaths'])

# Basic correlation
corr = df[['masfem','alldeaths','log_deaths','wind','min','category','ndam15']].corr()

# Regression models
# Model 1: log deaths ~ masfem
m1 = smf.ols('log_deaths ~ masfem', data=df).fit()
# Model 2: add storm severity controls
m2 = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit()
# Model 3: include damage as proxy for severity/exposure
m3 = smf.ols('log_deaths ~ masfem + wind + min + category + ndam15', data=df).fit()
# Model 4: use binary gender
m4 = smf.ols('log_deaths ~ gender_mf + wind + min + category', data=df).fit()

# Robustness: Poisson regression on deaths (count)
# add small constant to avoid issues with zero? Poisson handles zeros
m5 = smf.glm('alldeaths ~ masfem + wind + min + category', data=df, family=sm.families.Poisson()).fit()

# Extract results summary

def coef_info(model, term):
    if term not in model.params:
        return None
    return {
        'coef': float(model.params[term]),
        'se': float(model.bse[term]),
        'pval': float(model.pvalues[term])
    }

results = {
    'n': int(len(df)),
    'corr': corr.to_dict(),
    'm1': coef_info(m1, 'masfem'),
    'm1_r2': float(m1.rsquared),
    'm2': coef_info(m2, 'masfem'),
    'm2_r2': float(m2.rsquared),
    'm3': coef_info(m3, 'masfem'),
    'm3_r2': float(m3.rsquared),
    'm4': coef_info(m4, 'gender_mf'),
    'm4_r2': float(m4.rsquared),
    'm5': coef_info(m5, 'masfem'),
    'm5_aic': float(m5.aic),
}

with open('analysis_results.json','w') as f:
    json.dump(results,f,indent=2)

print(json.dumps(results,indent=2))
