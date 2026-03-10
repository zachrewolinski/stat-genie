import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename columns per info.json
mapping = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'masfem',
    'feature5': 'min_pressure',
    'feature6': 'female',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_since',
    'feature11': 'source',
    'feature12': 'masfem_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = df.rename(columns=mapping)

# Basic transformations

df['log_deaths'] = np.log1p(df['deaths'])
df['log_damage_2013'] = np.log1p(df['damage_2013'])

def ols_hc3(formula, data):
    return smf.ols(formula, data=data).fit(cov_type='HC3')

results = {}

# Bivariate OLS
m1 = ols_hc3('log_deaths ~ masfem', df)
results['bivariate_masfem'] = {
    'coef': float(m1.params['masfem']),
    'pvalue': float(m1.pvalues['masfem']),
    'ci_low': float(m1.conf_int().loc['masfem', 0]),
    'ci_high': float(m1.conf_int().loc['masfem', 1]),
    'n': int(m1.nobs),
}

# Controls for intensity and time
m2 = ols_hc3('log_deaths ~ masfem + min_pressure + max_wind + category + year', df)
results['controls_masfem'] = {
    'coef': float(m2.params['masfem']),
    'pvalue': float(m2.pvalues['masfem']),
    'ci_low': float(m2.conf_int().loc['masfem', 0]),
    'ci_high': float(m2.conf_int().loc['masfem', 1]),
    'n': int(m2.nobs),
}

m3 = ols_hc3('log_deaths ~ female + min_pressure + max_wind + category + year', df)
results['controls_female'] = {
    'coef': float(m3.params['female']),
    'pvalue': float(m3.pvalues['female']),
    'ci_low': float(m3.conf_int().loc['female', 0]),
    'ci_high': float(m3.conf_int().loc['female', 1]),
    'n': int(m3.nobs),
}

# Add damage control
m4 = ols_hc3('log_deaths ~ masfem + min_pressure + max_wind + category + log_damage_2013 + year', df)
results['controls_damage_masfem'] = {
    'coef': float(m4.params['masfem']),
    'pvalue': float(m4.pvalues['masfem']),
    'ci_low': float(m4.conf_int().loc['masfem', 0]),
    'ci_high': float(m4.conf_int().loc['masfem', 1]),
    'n': int(m4.nobs),
}

# Interaction with severity (category)
# Center variables to reduce collinearity

df['masfem_c'] = df['masfem'] - df['masfem'].mean()
df['category_c'] = df['category'] - df['category'].mean()

m5 = ols_hc3('log_deaths ~ masfem_c * category_c + min_pressure + max_wind + year', df)
results['interaction_masfem_category'] = {
    'coef_interaction': float(m5.params['masfem_c:category_c']),
    'pvalue_interaction': float(m5.pvalues['masfem_c:category_c']),
    'ci_low': float(m5.conf_int().loc['masfem_c:category_c', 0]),
    'ci_high': float(m5.conf_int().loc['masfem_c:category_c', 1]),
    'n': int(m5.nobs),
}

# GLM Negative Binomial on deaths counts
# Use log link; add small constant not needed for count

try:
    nb = smf.glm('deaths ~ masfem + min_pressure + max_wind + category + year',
                 data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    results['nb_masfem'] = {
        'coef': float(nb.params['masfem']),
        'pvalue': float(nb.pvalues['masfem']),
        'ci_low': float(nb.conf_int().loc['masfem', 0]),
        'ci_high': float(nb.conf_int().loc['masfem', 1]),
        'n': int(nb.nobs),
    }
except Exception as e:
    results['nb_masfem'] = {'error': str(e)}

# Simple correlations
results['corr_masfem_deaths'] = float(df['masfem'].corr(df['deaths']))
results['corr_female_deaths'] = float(df['female'].corr(df['deaths']))

# Print results
for k, v in results.items():
    print(k, v)
