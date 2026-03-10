import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')

# Variable mapping based on info.json descriptions
# femininity ratings
fem_rating = 'category'       # 1-11 masculinity-femininity index (coders)
fem_rating_mturk = 'ind'      # MTurk ratings 1-11
fem_binary = 'masfem_mturk'   # 0 male, 1 female

# outcomes (proxies for precautionary measures)
deaths = 'name'               # total deaths

damage = 'elapsedyrs'         # normalized damage (2013 dollars)

# severity controls
max_wind = 'year'             # max wind speed
min_pressure = 'ndam15'       # min pressure
category_ss = 'gender_mf'     # Saffir-Simpson category
storm_year = 'wind'           # year of hurricane

# Prepare data
for col in [deaths, damage, max_wind, min_pressure, category_ss, storm_year, fem_rating, fem_rating_mturk, fem_binary]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Log transforms for skewed outcomes
for outcome, newcol in [(deaths, 'log_deaths'), (damage, 'log_damage')]:
    df[newcol] = np.log1p(df[outcome])

# Basic correlations
corrs = {}
for fem in [fem_rating, fem_rating_mturk, fem_binary]:
    corrs[fem] = {
        'pearson_log_deaths': df[[fem, 'log_deaths']].corr().iloc[0,1],
        'spearman_log_deaths': df[[fem, 'log_deaths']].corr(method='spearman').iloc[0,1],
        'pearson_log_damage': df[[fem, 'log_damage']].corr().iloc[0,1],
        'spearman_log_damage': df[[fem, 'log_damage']].corr(method='spearman').iloc[0,1],
    }

print('Correlations')
for fem, vals in corrs.items():
    print(fem, vals)

# Regression models
models = {}

# Simple models
models['deaths_simple_fem'] = smf.ols(f"log_deaths ~ {fem_rating}", data=df).fit(cov_type='HC3')
models['deaths_simple_binary'] = smf.ols(f"log_deaths ~ {fem_binary}", data=df).fit(cov_type='HC3')

# Severity-controlled model
models['deaths_controls_fem'] = smf.ols(
    f"log_deaths ~ {fem_rating} + {max_wind} + {min_pressure} + {category_ss} + {storm_year}",
    data=df
).fit(cov_type='HC3')

models['deaths_controls_binary'] = smf.ols(
    f"log_deaths ~ {fem_binary} + {max_wind} + {min_pressure} + {category_ss} + {storm_year}",
    data=df
).fit(cov_type='HC3')

# Alternative femininity measure
models['deaths_controls_mturk'] = smf.ols(
    f"log_deaths ~ {fem_rating_mturk} + {max_wind} + {min_pressure} + {category_ss} + {storm_year}",
    data=df
).fit(cov_type='HC3')

# Damage outcome as secondary proxy
models['damage_controls_fem'] = smf.ols(
    f"log_damage ~ {fem_rating} + {max_wind} + {min_pressure} + {category_ss} + {storm_year}",
    data=df
).fit(cov_type='HC3')

print('\nKey coefficients')
for name, model in models.items():
    params = model.params
    pvals = model.pvalues
    for var in [fem_rating, fem_binary, fem_rating_mturk]:
        if var in params.index:
            print(name, var, 'coef', params[var], 'p', pvals[var])

# Save key stats to json-ish print for review
print('\nModel summaries (fem coefficients)')
for name, model in models.items():
    for var in [fem_rating, fem_binary, fem_rating_mturk]:
        if var in model.params.index:
            print(name, var, 'coef', model.params[var], 'p', model.pvalues[var], 'n', int(model.nobs))
