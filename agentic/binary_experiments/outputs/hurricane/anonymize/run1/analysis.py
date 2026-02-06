import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = '/home/chenwang/stat-genie/agentic/experiments/outputs/hurricane/anonymize/run1/hurricane.csv'
df = pd.read_csv(path)

# Rename columns for clarity
cols = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'masfem_index',
    'feature5': 'min_pressure',
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_elapsed',
    'feature11': 'source',
    'feature12': 'mturk_masfem',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = df.rename(columns=cols)

# Outcome: log1p deaths to reduce skew
# Controls: storm intensity and exposure proxies
analysis_df = df.copy()
analysis_df['log_deaths'] = np.log1p(analysis_df['deaths'])
analysis_df['log_damage_2015'] = np.log1p(analysis_df['damage_2015'])

# Drop rows with missing values in model variables
model_vars = ['log_deaths', 'masfem_index', 'female_binary', 'max_wind', 'min_pressure',
              'category', 'log_damage_2015', 'year']
analysis_df = analysis_df.dropna(subset=model_vars)

# Model 1: continuous masculinity-femininity index
X1 = analysis_df[['masfem_index', 'max_wind', 'min_pressure', 'category', 'log_damage_2015', 'year']]
X1 = sm.add_constant(X1)
model1 = sm.OLS(analysis_df['log_deaths'], X1).fit(cov_type='HC3')

# Model 2: binary female indicator
X2 = analysis_df[['female_binary', 'max_wind', 'min_pressure', 'category', 'log_damage_2015', 'year']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(analysis_df['log_deaths'], X2).fit(cov_type='HC3')

# Print summaries for inspection
print('Model 1 (masfem_index):')
print(model1.summary())
print('\nModel 2 (female_binary):')
print(model2.summary())

# Simple decision rule for conclusion
masfem_coef = model1.params['masfem_index']
masfem_p = model1.pvalues['masfem_index']

female_coef = model2.params['female_binary']
female_p = model2.pvalues['female_binary']

print('\nKey results:')
print(f'masfem_index coef={masfem_coef:.4f}, p={masfem_p:.4f}')
print(f'female_binary coef={female_coef:.4f}, p={female_p:.4f}')
