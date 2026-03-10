import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('hurricane.csv')

# Relevant columns
# Handle missing values by dropping rows with missing in any relevant columns

# Create outcome
df['log_deaths'] = np.log1p(df['alldeaths'])

# Pearson correlation
corr_masfem = stats.pearsonr(df['masfem'], df['log_deaths'])
corr_gender = stats.pearsonr(df['gender_mf'], df['log_deaths'])

# Define function to run OLS with robust SE
def run_ols(y, X, label):
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')
    return label, model

# Model A: simple
label_a, model_a = run_ols(df['log_deaths'], df[['masfem']], 'A: log_deaths ~ masfem')

# Model B: with controls
controls = ['masfem', 'category', 'wind', 'min', 'ndam15', 'year']
label_b, model_b = run_ols(df['log_deaths'], df[controls], 'B: log_deaths ~ masfem + category + wind + min + ndam15 + year')

# Model C: binary gender
label_c, model_c = run_ols(df['log_deaths'], df[['gender_mf']], 'C: log_deaths ~ gender_mf')

# Model D: gender + controls
controls_gender = ['gender_mf', 'category', 'wind', 'min', 'ndam15', 'year']
label_d, model_d = run_ols(df['log_deaths'], df[controls_gender], 'D: log_deaths ~ gender_mf + category + wind + min + ndam15 + year')

# Summaries to print
print('N rows:', len(df))
print('Correlation masfem vs log_deaths:', corr_masfem)
print('Correlation gender_mf vs log_deaths:', corr_gender)

for label, model in [(label_a, model_a), (label_b, model_b), (label_c, model_c), (label_d, model_d)]:
    coef = model.params
    pvals = model.pvalues
    print('\n', label)
    print(model.summary().tables[1])
    # Print focal coefficient
    if 'masfem' in coef:
        print('masfem coef:', coef['masfem'], 'p:', pvals['masfem'])
    if 'gender_mf' in coef:
        print('gender_mf coef:', coef['gender_mf'], 'p:', pvals['gender_mf'])
