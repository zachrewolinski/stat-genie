import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

pd.set_option('display.max_columns', None)

# Load data

df = pd.read_csv('hurricane.csv')

# Basic derived variables

df['log_deaths'] = np.log1p(df['alldeaths'])

# Summary stats
print('Rows', len(df))
print('Deaths summary', df['alldeaths'].describe())
print('Masfem summary', df['masfem'].describe())

# Correlations
for col in ['masfem','masfem_mturk','gender_mf']:
    if col in df.columns:
        print('corr log_deaths vs', col, df['log_deaths'].corr(df[col]))

# OLS with controls
models = {}
formulas = {
    'ols_basic': 'log_deaths ~ masfem',
    'ols_controls': 'log_deaths ~ masfem + wind + min + category',
    'ols_controls_year': 'log_deaths ~ masfem + wind + min + category + year',
    'ols_mturk_controls': 'log_deaths ~ masfem_mturk + wind + min + category',
    'ols_gender_controls': 'log_deaths ~ gender_mf + wind + min + category',
}

for name, formula in formulas.items():
    model = smf.ols(formula, data=df).fit(cov_type='HC3')
    models[name] = model
    print('\n', name)
    print(model.summary().tables[1])

# Negative binomial

nb_model = smf.glm('alldeaths ~ masfem + wind + min + category', data=df, family=sm.families.NegativeBinomial()).fit()
print('\nnegative_binomial')
print(nb_model.summary().tables[1])

# Check dispersion of Poisson (overdispersion indicator)

poisson = smf.glm('alldeaths ~ masfem + wind + min + category', data=df, family=sm.families.Poisson()).fit()
pearson_chi2 = sum(poisson.resid_pearson**2)
print('\npoisson dispersion', pearson_chi2/poisson.df_resid)

# Simple robustness: replace masfem with masfem_mturk in NB
nb_mturk = smf.glm('alldeaths ~ masfem_mturk + wind + min + category', data=df, family=sm.families.NegativeBinomial()).fit()
print('\nnegative_binomial_mturk')
print(nb_mturk.summary().tables[1])

# See if effect sign is positive/negative

def extract_effect(model, var):
    coef = model.params[var]
    se = model.bse[var]
    p = model.pvalues[var]
    return coef, se, p

for name, model in models.items():
    var = 'masfem' if 'mturk' not in name and 'gender' not in name else ('masfem_mturk' if 'mturk' in name else 'gender_mf')
    if var in model.params.index:
        coef, se, p = extract_effect(model, var)
        print(f'{name} {var} coef={coef:.4f} se={se:.4f} p={p:.4f}')

coef, se, p = extract_effect(nb_model, 'masfem')
print(f'nb_model masfem coef={coef:.4f} se={se:.4f} p={p:.4f}')
