import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns to meaningful names based on info.json metadata
# Column meanings are described in info.json; names are shuffled.
renamed = {
    'ndam': 'storm_id',
    'wind': 'year',
    'alldeaths': 'storm_name',
    'category': 'fem_rating_coders',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'fem_binary',
    'gender_mf': 'ss_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'years_elapsed',
    'min': 'source',
    'ind': 'fem_rating_mturk',
    'year': 'wind_speed',
    'source': 'damage_2015',
}

df = df.rename(columns=renamed)

# Basic sanity
n = len(df)

# Derived variables
# log deaths to reduce skew

df['log_deaths'] = np.log1p(df['deaths'])

# Summary comparisons: female vs male
female = df[df['fem_binary'] == 1]
male = df[df['fem_binary'] == 0]

summary = {
    'n': n,
    'female_n': len(female),
    'male_n': len(male),
    'female_mean_deaths': female['deaths'].mean(),
    'male_mean_deaths': male['deaths'].mean(),
    'female_median_deaths': female['deaths'].median(),
    'male_median_deaths': male['deaths'].median(),
}

# Correlations
corrs = {}
for col in ['fem_rating_coders', 'fem_rating_mturk']:
    corrs[col] = {
        'pearson_log_deaths': df[[col, 'log_deaths']].corr().iloc[0,1],
        'spearman_log_deaths': df[[col, 'log_deaths']].corr(method='spearman').iloc[0,1],
        'pearson_deaths': df[[col, 'deaths']].corr().iloc[0,1],
        'spearman_deaths': df[[col, 'deaths']].corr(method='spearman').iloc[0,1],
    }

# Regression: log(deaths+1) ~ fem_rating + controls for storm severity + year
# Use wind_speed and min_pressure as severity proxies; include ss_category optionally.

# Model with coder rating
formula_coder = 'log_deaths ~ fem_rating_coders + wind_speed + min_pressure + ss_category + year'
model_coder = smf.ols(formula_coder, data=df).fit(cov_type='HC3')

# Model with MTurk rating
formula_mturk = 'log_deaths ~ fem_rating_mturk + wind_speed + min_pressure + ss_category + year'
model_mturk = smf.ols(formula_mturk, data=df).fit(cov_type='HC3')

# Model with binary female indicator
formula_bin = 'log_deaths ~ fem_binary + wind_speed + min_pressure + ss_category + year'
model_bin = smf.ols(formula_bin, data=df).fit(cov_type='HC3')

results = {
    'coder': {
        'coef': model_coder.params['fem_rating_coders'],
        'pvalue': model_coder.pvalues['fem_rating_coders'],
    },
    'mturk': {
        'coef': model_mturk.params['fem_rating_mturk'],
        'pvalue': model_mturk.pvalues['fem_rating_mturk'],
    },
    'binary': {
        'coef': model_bin.params['fem_binary'],
        'pvalue': model_bin.pvalues['fem_binary'],
    }
}

print('SUMMARY', summary)
print('CORRS', corrs)
print('REGRESSION', results)

