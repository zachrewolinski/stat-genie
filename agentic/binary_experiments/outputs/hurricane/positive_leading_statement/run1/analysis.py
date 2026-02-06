import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns are numeric
numeric_cols = ['masfem', 'masfem_mturk', 'alldeaths', 'wind', 'min', 'category', 'ndam15', 'ndam', 'elapsedyrs']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Outcome: log deaths (add 1 to handle zeros)
df['log_deaths'] = np.log1p(df['alldeaths'])

# Primary model: femininity vs log deaths, controlling for hurricane strength
# Controls chosen to capture storm intensity: wind speed, minimum pressure, and category
model = smf.ols('log_deaths ~ masfem + wind + min + category', data=df).fit(cov_type='HC3')

# Alternate model using MTurk femininity ratings
model_mturk = smf.ols('log_deaths ~ masfem_mturk + wind + min + category', data=df).fit(cov_type='HC3')

# Simple comparison: mean deaths by gender_mf
mean_deaths_by_gender = df.groupby('gender_mf')['alldeaths'].mean()

# Correlation between femininity and deaths (log)
cor_masfem = df['masfem'].corr(df['log_deaths'])
cor_mturk = df['masfem_mturk'].corr(df['log_deaths'])

print('Primary OLS (HC3 robust SE)')
print(model.summary())
print('\nAlternate OLS (MTurk femininity, HC3 robust SE)')
print(model_mturk.summary())
print('\nMean deaths by gender_mf (0=male, 1=female)')
print(mean_deaths_by_gender)
print('\nCorrelation (masfem vs log deaths):', cor_masfem)
print('Correlation (masfem_mturk vs log deaths):', cor_mturk)

# Save key results for interpretation
results = {
    'coef_masfem': model.params.get('masfem', np.nan),
    'p_masfem': model.pvalues.get('masfem', np.nan),
    'coef_masfem_mturk': model_mturk.params.get('masfem_mturk', np.nan),
    'p_masfem_mturk': model_mturk.pvalues.get('masfem_mturk', np.nan),
    'mean_deaths_male': mean_deaths_by_gender.get(0, np.nan),
    'mean_deaths_female': mean_deaths_by_gender.get(1, np.nan),
    'cor_masfem': cor_masfem,
    'cor_masfem_mturk': cor_mturk,
}

pd.Series(results).to_csv('analysis_results_summary.csv')
