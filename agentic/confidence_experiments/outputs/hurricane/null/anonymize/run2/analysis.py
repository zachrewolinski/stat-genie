import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'femininity',
    'feature5': 'min_pressure',
    'feature6': 'female_name',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_since',
    'feature11': 'source',
    'feature12': 'femininity_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015'
})

# Basic transforms
_df['log_deaths'] = np.log1p(_df['deaths'])

# Clean: ensure no missing
_df = _df.dropna(subset=['log_deaths', 'femininity', 'female_name', 'category', 'min_pressure', 'max_wind', 'year'])

# Models
results = {}

# Simple bivariate
results['bivariate_fem'] = smf.ols('log_deaths ~ femininity', data=_df).fit(cov_type='HC3')
results['bivariate_female'] = smf.ols('log_deaths ~ female_name', data=_df).fit(cov_type='HC3')

# Multivariate with storm severity controls
results['multivar_fem'] = smf.ols('log_deaths ~ femininity + category + min_pressure + max_wind + year', data=_df).fit(cov_type='HC3')
results['multivar_female'] = smf.ols('log_deaths ~ female_name + category + min_pressure + max_wind + year', data=_df).fit(cov_type='HC3')

# Alternative: use femininity_mturk instead of coder ratings
results['multivar_fem_mturk'] = smf.ols('log_deaths ~ femininity_mturk + category + min_pressure + max_wind + year', data=_df).fit(cov_type='HC3')

# Output key results
summary_rows = []
for key, res in results.items():
    if 'femininity_mturk' in res.params:
        coef = res.params['femininity_mturk']
        pval = res.pvalues['femininity_mturk']
    elif 'femininity' in res.params:
        coef = res.params['femininity']
        pval = res.pvalues['femininity']
    elif 'female_name' in res.params:
        coef = res.params['female_name']
        pval = res.pvalues['female_name']
    else:
        coef = np.nan
        pval = np.nan
    summary_rows.append((key, coef, pval, res.rsquared))

summary = pd.DataFrame(summary_rows, columns=['model', 'coef', 'pval', 'r2'])

print('N:', len(_df))
print(summary)

# Additional: Spearman correlations (nonparametric)
from scipy.stats import spearmanr

rho_fem, p_fem = spearmanr(_df['femininity'], _df['deaths'])
rho_female, p_female = spearmanr(_df['female_name'], _df['deaths'])
print('Spearman femininity vs deaths:', rho_fem, p_fem)
print('Spearman female_name vs deaths:', rho_female, p_female)

# Print key coefficients for multivariate models
for key in ['multivar_fem', 'multivar_female', 'multivar_fem_mturk']:
    res = results[key]
    print('\n', key)
    print(res.summary().tables[1])
