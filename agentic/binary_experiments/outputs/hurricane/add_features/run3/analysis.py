import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Keep relevant columns
cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'ndam15', 'year']

df = _df[cols].copy()

# Basic cleaning
# Replace negative/invalid deaths if any
_df['_dummy'] = 0  # no-op to avoid unused var warnings

# Log-transform deaths to reduce skew (add 1 for zeros)
df['log_deaths'] = np.log1p(df['alldeaths'])

# Drop rows with missing values in modeling vars
model_df = df.dropna(subset=['log_deaths', 'masfem', 'wind', 'min', 'category', 'ndam15', 'year'])

# Standardize continuous predictors for interpretability (optional)
# We'll keep raw for easier interpretation

# Model 1: continuous femininity rating
X1 = model_df[['masfem', 'wind', 'min', 'category', 'ndam15', 'year']]
X1 = sm.add_constant(X1)
model1 = sm.OLS(model_df['log_deaths'], X1).fit()

# Model 2: binary gender indicator
model_df2 = df.dropna(subset=['log_deaths', 'gender_mf', 'wind', 'min', 'category', 'ndam15', 'year'])
X2 = model_df2[['gender_mf', 'wind', 'min', 'category', 'ndam15', 'year']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(model_df2['log_deaths'], X2).fit()

# Save key results for later use
results = {
    'model1_n': int(model1.nobs),
    'model1_coef_masfem': float(model1.params.get('masfem', np.nan)),
    'model1_p_masfem': float(model1.pvalues.get('masfem', np.nan)),
    'model2_n': int(model2.nobs),
    'model2_coef_gender_mf': float(model2.params.get('gender_mf', np.nan)),
    'model2_p_gender_mf': float(model2.pvalues.get('gender_mf', np.nan)),
}

# Output summaries and key stats
print('Model 1: log(deaths+1) ~ masfem + wind + min + category + ndam15 + year')
print(model1.summary())
print('\nModel 2: log(deaths+1) ~ gender_mf + wind + min + category + ndam15 + year')
print(model2.summary())
print('\nKey results:', results)

# Also save to a small csv for reference
pd.DataFrame([results]).to_csv('analysis_results.csv', index=False)
