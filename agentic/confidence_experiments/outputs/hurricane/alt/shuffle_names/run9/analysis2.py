import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')

# Infer variable meanings based on ranges and prior dataset knowledge
# deaths: column 'name' (0-1833)
# femininity index: 'category' (1-11 scale) and 'ind' (1-11 scale)
# storm category: 'gender_mf' (1-5)
# max wind speed: 'year' (75-190 mph)
# min pressure: 'ndam15' (909-1002 mb)
# year of hurricane: 'wind' (1950-2012)

# Prepare variables
analysis = df.copy()
analysis['log_deaths'] = np.log1p(analysis['name'])

# Simple correlations
corr_category = analysis['category'].corr(analysis['log_deaths'])
corr_ind = analysis['ind'].corr(analysis['log_deaths'])

print('Correlation (category femininity vs log deaths):', corr_category)
print('Correlation (ind femininity vs log deaths):', corr_ind)

# OLS with controls (robust SE)
formula = 'log_deaths ~ category + gender_mf + year + ndam15 + wind'
model = smf.ols(formula, data=analysis).fit(cov_type='HC3')
print('\nOLS with category femininity:')
print(model.summary())

formula2 = 'log_deaths ~ ind + gender_mf + year + ndam15 + wind'
model2 = smf.ols(formula2, data=analysis).fit(cov_type='HC3')
print('\nOLS with ind femininity:')
print(model2.summary())

# Check coefficient and p-value for femininity predictors
for label, mdl, var in [('category', model, 'category'), ('ind', model2, 'ind')]:
    coef = mdl.params[var]
    pval = mdl.pvalues[var]
    print(f"\n{label} coef={coef:.4f}, p={pval:.4f}")
