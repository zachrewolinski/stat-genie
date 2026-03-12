import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

# Rename for readability
rename_map = {
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'masfem_coder',
    'feature5': 'min_pressure',
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_since',
    'feature11': 'source',
    'feature12': 'masfem_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}


df = df.rename(columns=rename_map)

# Outcome: fatalities. Use log transform for skewness.
df['log_deaths'] = np.log1p(df['deaths'])

# Core severity covariates
covariates = ['category', 'min_pressure', 'max_wind', 'year']

# Ensure no missing in the required columns
model_df = df[['log_deaths', 'deaths', 'masfem_coder', 'masfem_mturk', 'female_binary'] + covariates].dropna()

results = {}

# Model 1: coder femininity index
formula1 = 'log_deaths ~ masfem_coder + category + min_pressure + max_wind + year'
model1 = smf.ols(formula1, data=model_df).fit(cov_type='HC3')

# Model 2: MTurk femininity index
formula2 = 'log_deaths ~ masfem_mturk + category + min_pressure + max_wind + year'
model2 = smf.ols(formula2, data=model_df).fit(cov_type='HC3')

# Model 3: binary female indicator
formula3 = 'log_deaths ~ female_binary + category + min_pressure + max_wind + year'
model3 = smf.ols(formula3, data=model_df).fit(cov_type='HC3')

# Simple bivariate correlations (Spearman) for robustness to skew
spearman_coder = model_df[['masfem_coder', 'deaths']].corr(method='spearman').iloc[0,1]
spearman_mturk = model_df[['masfem_mturk', 'deaths']].corr(method='spearman').iloc[0,1]

results['n'] = int(model_df.shape[0])
results['spearman_coder_deaths'] = float(spearman_coder)
results['spearman_mturk_deaths'] = float(spearman_mturk)

# Extract key stats
for label, model, var in [
    ('coder', model1, 'masfem_coder'),
    ('mturk', model2, 'masfem_mturk'),
    ('binary', model3, 'female_binary'),
]:
    results[label] = {
        'coef': float(model.params[var]),
        'pvalue': float(model.pvalues[var]),
        'ci_low': float(model.conf_int().loc[var, 0]),
        'ci_high': float(model.conf_int().loc[var, 1]),
    }

# Save results for inspection
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
