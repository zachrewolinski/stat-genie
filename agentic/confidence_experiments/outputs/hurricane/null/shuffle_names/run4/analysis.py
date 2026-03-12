import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map variables based on info.json descriptions
# names are shuffled; these are the actual meanings
var_map = {
    'deaths': 'name',            # total deaths
    'femininity': 'category',    # 1-11 masculinity-femininity index
    'fem_binary': 'masfem_mturk',# 0 male, 1 female
    'year': 'wind',              # year of hurricane
    'wind_speed': 'year',        # max wind speed at landfall
    'min_pressure': 'ndam15',    # minimum pressure at landfall
    'ssf_category': 'gender_mf', # Saffir-Simpson category (1-5)
    'damage_2013': 'elapsedyrs', # damage normalized to 2013 dollars
    'damage_2015': 'source',     # damage normalized to 2015 dollars
}

# Create analysis dataframe
cols = list(var_map.values())
sub = df[cols].copy()

# log-transform skewed variables
sub['log_deaths'] = np.log1p(sub[var_map['deaths']])
sub['log_damage_2015'] = np.log1p(sub[var_map['damage_2015']])

# Simple correlations
pearson_r, pearson_p = stats.pearsonr(sub[var_map['femininity']], sub[var_map['deaths']])
spearman_r, spearman_p = stats.spearmanr(sub[var_map['femininity']], sub[var_map['deaths']])

# OLS models with robust SE

def fit_ols(y, X):
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model

# Model 1: log deaths ~ femininity
m1 = fit_ols(sub['log_deaths'], sub[[var_map['femininity']]])

# Model 2: log deaths ~ femininity + controls
controls = [
    var_map['wind_speed'],
    var_map['min_pressure'],
    var_map['ssf_category'],
    'log_damage_2015',
    var_map['year'],
]

m2 = fit_ols(sub['log_deaths'], sub[[var_map['femininity']] + controls])

# Model 3: log deaths ~ fem_binary + controls
m3 = fit_ols(sub['log_deaths'], sub[[var_map['fem_binary']] + controls])

# Collect key results
results = {
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'spearman_r': spearman_r,
    'spearman_p': spearman_p,
    'm1_coef': m1.params[var_map['femininity']],
    'm1_p': m1.pvalues[var_map['femininity']],
    'm2_coef': m2.params[var_map['femininity']],
    'm2_p': m2.pvalues[var_map['femininity']],
    'm3_coef': m3.params[var_map['fem_binary']],
    'm3_p': m3.pvalues[var_map['fem_binary']],
    'm2_n': int(m2.nobs),
    'm3_n': int(m3.nobs),
    'm2_r2': m2.rsquared,
    'm3_r2': m3.rsquared,
}

print(results)

# Also show short summaries for context
print('\nM2 summary (femininity + controls):')
print(m2.summary())
print('\nM3 summary (binary female + controls):')
print(m3.summary())
