import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns based on observed values
# Existing columns:
# ndam (id), wind (year), alldeaths (name), category (femininity rating 1-11),
# ndam15 (min pressure), masfem_mturk (binary female), gender_mf (Saffir-Simpson category 1-5),
# name (deaths), elapsedyrs (normalized damage), masfem (years since 2013),
# min (source), ind (mturk rating 1-11), year (max wind speed), source (raw damage)

# Ensure numeric columns parsed correctly
numeric_cols = [
    'ndam', 'wind', 'category', 'ndam15', 'masfem_mturk', 'gender_mf',
    'name', 'elapsedyrs', 'masfem', 'ind', 'year', 'source'
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Rename to clearer analysis names
renamed = df.rename(columns={
    'wind': 'year_hurr',
    'alldeaths': 'hurr_name',
    'category': 'fem_rating',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'ss_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_norm',
    'masfem': 'years_since_2013',
    'ind': 'fem_mturk',
    'year': 'max_wind',
    'source': 'damage_raw'
})

# Outcome and predictors
# Use log1p of deaths due to skew
renamed['log_deaths'] = np.log1p(renamed['deaths'])

# Build model dataframe
model_df = renamed[[
    'log_deaths', 'fem_rating', 'female_binary', 'max_wind',
    'min_pressure', 'ss_category', 'year_hurr'
]].dropna()

# Standardize continuous predictors for interpretability (optional)
# We keep unstandardized for straightforward coefficient/p-value; interaction in original scale

# Model 1: main effects with fem_rating
X1 = model_df[['fem_rating', 'max_wind', 'min_pressure', 'ss_category', 'year_hurr']]
X1 = sm.add_constant(X1)
model1 = sm.OLS(model_df['log_deaths'], X1).fit(cov_type='HC3')

# Model 2: add interaction fem_rating * max_wind
model_df['fem_x_wind'] = model_df['fem_rating'] * model_df['max_wind']
X2 = model_df[['fem_rating', 'max_wind', 'fem_x_wind', 'min_pressure', 'ss_category', 'year_hurr']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(model_df['log_deaths'], X2).fit(cov_type='HC3')

# Model 3: use binary female indicator
X3 = model_df[['female_binary', 'max_wind', 'min_pressure', 'ss_category', 'year_hurr']]
X3 = sm.add_constant(X3)
model3 = sm.OLS(model_df['log_deaths'], X3).fit(cov_type='HC3')

# Extract key stats
results = {
    'n': int(model_df.shape[0]),
    'model1': {
        'coef_fem_rating': model1.params.get('fem_rating', np.nan),
        'p_fem_rating': model1.pvalues.get('fem_rating', np.nan)
    },
    'model2': {
        'coef_fem_rating': model2.params.get('fem_rating', np.nan),
        'p_fem_rating': model2.pvalues.get('fem_rating', np.nan),
        'coef_fem_x_wind': model2.params.get('fem_x_wind', np.nan),
        'p_fem_x_wind': model2.pvalues.get('fem_x_wind', np.nan)
    },
    'model3': {
        'coef_female_binary': model3.params.get('female_binary', np.nan),
        'p_female_binary': model3.pvalues.get('female_binary', np.nan)
    }
}

# Also compute simple correlations between fem_rating and deaths/log_deaths
corr = renamed[['fem_rating', 'deaths', 'log_deaths']].corr(numeric_only=True)
results['corr_fem_deaths'] = float(corr.loc['fem_rating', 'deaths'])
results['corr_fem_log_deaths'] = float(corr.loc['fem_rating', 'log_deaths'])

# Save numeric summary for reference
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
