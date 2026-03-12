import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename columns to clearer names based on metadata inspection
renamed = df.rename(columns={
    'wind': 'year',                 # actual year
    'alldeaths': 'storm_name',
    'category': 'fem_rating',       # 1-11 femininity index
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'ss_category',     # Saffir-Simpson category
    'name': 'deaths',               # total deaths
    'elapsedyrs': 'damage_2013',    # normalized damage (2013)
    'masfem': 'years_elapsed',
    'ind': 'fem_rating_mturk',
    'year': 'max_wind',             # maximum wind speed
    'source': 'damage_2015',        # normalized damage (2015)
})

# Create analysis variables
renamed['log_deaths'] = np.log1p(renamed['deaths'])
renamed['log_damage_2013'] = np.log1p(renamed['damage_2013'])
renamed['log_damage_2015'] = np.log1p(renamed['damage_2015'])

# Drop rows with missing values for model variables
model_vars = ['log_deaths', 'fem_rating', 'female_binary', 'max_wind', 'min_pressure',
              'ss_category', 'log_damage_2013', 'year']
model_df = renamed.dropna(subset=model_vars).copy()

# Model with femininity rating (primary)
formula_fem = 'log_deaths ~ fem_rating + max_wind + min_pressure + ss_category + log_damage_2013 + year'
model_fem = smf.ols(formula_fem, data=model_df).fit(cov_type='HC3')

# Model with binary female indicator
formula_bin = 'log_deaths ~ female_binary + max_wind + min_pressure + ss_category + log_damage_2013 + year'
model_bin = smf.ols(formula_bin, data=model_df).fit(cov_type='HC3')

# Simple correlation (Spearman) between fem_rating and deaths
spearman = renamed[['fem_rating', 'deaths']].corr(method='spearman').iloc[0, 1]

# Collect results
results = {
    'n': int(model_df.shape[0]),
    'fem_rating_coef': float(model_fem.params['fem_rating']),
    'fem_rating_p': float(model_fem.pvalues['fem_rating']),
    'female_binary_coef': float(model_bin.params['female_binary']),
    'female_binary_p': float(model_bin.pvalues['female_binary']),
    'spearman_fem_deaths': float(spearman),
}

print(results)
