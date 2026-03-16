import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic prep
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Controls for intensity
controls = ['category', 'wind', 'min', 'year']

# OLS on log deaths with masfem
formula_ols = 'log_deaths ~ masfem + ' + ' + '.join(controls)
ols = smf.ols(formula_ols, data=_df).fit(cov_type='HC3')

# OLS on log deaths with masfem_mturk
formula_ols_mturk = 'log_deaths ~ masfem_mturk + ' + ' + '.join(controls)
ols_mturk = smf.ols(formula_ols_mturk, data=_df).fit(cov_type='HC3')

# OLS with gender_mf instead of masfem
formula_ols_gender = 'log_deaths ~ gender_mf + ' + ' + '.join(controls)
ols_gender = smf.ols(formula_ols_gender, data=_df).fit(cov_type='HC3')

# Poisson regression on deaths (count)
formula_pois = 'alldeaths ~ masfem + ' + ' + '.join(controls)
pois = smf.glm(formula_pois, data=_df, family=sm.families.Poisson()).fit(cov_type='HC3')

# Poisson dispersion diagnostic (Pearson chi2 / df)
pearson_chi2 = sum(pois.resid_pearson ** 2)
dispersion = pearson_chi2 / pois.df_resid

# Negative binomial regression (NB2, estimates alpha)
try:
    nb2 = smf.negativebinomial(formula_pois, data=_df).fit(disp=0)
except Exception as e:
    nb2 = None

# Negative binomial with masfem_mturk
formula_nb_mturk = 'alldeaths ~ masfem_mturk + ' + ' + '.join(controls)
try:
    nb2_mturk = smf.negativebinomial(formula_nb_mturk, data=_df).fit(disp=0)
except Exception as e:
    nb2_mturk = None

# Interaction: femininity x intensity (category)
formula_inter = 'log_deaths ~ masfem * category + wind + min + year'
ols_inter = smf.ols(formula_inter, data=_df).fit(cov_type='HC3')

# Correlations
corr = _df[['masfem', 'masfem_mturk', 'alldeaths', 'log_deaths', 'category', 'wind', 'min', 'year']].corr(numeric_only=True)

print('OLS log deaths with masfem + controls')
print(ols.summary().tables[1])
print('\nOLS log deaths with masfem_mturk + controls')
print(ols_mturk.summary().tables[1])
print('\nOLS log deaths with gender_mf + controls')
print(ols_gender.summary().tables[1])
print('\nPoisson deaths with masfem + controls')
print(pois.summary().tables[1])
print('\nPoisson dispersion (Pearson chi2 / df):', dispersion)

if nb2 is not None:
    print('\nNegBin (NB2) deaths with masfem + controls')
    print(nb2.summary().tables[1])

if nb2_mturk is not None:
    print('\nNegBin (NB2) deaths with masfem_mturk + controls')
    print(nb2_mturk.summary().tables[1])

print('\nOLS interaction masfem*category')
print(ols_inter.summary().tables[1])

print('\nCorrelations (selected)')
print(corr)

# Key stats for reporting
print('\nKey coefficients:')
print('OLS masfem coef:', ols.params['masfem'], 'p=', ols.pvalues['masfem'])
print('OLS masfem_mturk coef:', ols_mturk.params['masfem_mturk'], 'p=', ols_mturk.pvalues['masfem_mturk'])
print('OLS gender_mf coef:', ols_gender.params['gender_mf'], 'p=', ols_gender.pvalues['gender_mf'])
print('Poisson masfem coef:', pois.params['masfem'], 'p=', pois.pvalues['masfem'])
if nb2 is not None:
    print('NegBin NB2 masfem coef:', nb2.params['masfem'], 'p=', nb2.pvalues['masfem'])
if nb2_mturk is not None:
    print('NegBin NB2 masfem_mturk coef:', nb2_mturk.params['masfem_mturk'], 'p=', nb2_mturk.pvalues['masfem_mturk'])
print('OLS interaction masfem:category coef:', ols_inter.params.get('masfem:category'), 'p=', ols_inter.pvalues.get('masfem:category'))
