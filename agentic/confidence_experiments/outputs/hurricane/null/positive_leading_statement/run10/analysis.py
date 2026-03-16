import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
DATA_PATH = 'hurricane.csv'

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure numeric columns are numeric
numeric_cols = [
    'masfem', 'gender_mf', 'min', 'category', 'alldeaths', 'ndam', 'elapsedyrs', 'masfem_mturk', 'wind', 'ndam15', 'year'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create log deaths
# log1p to handle zeros
if 'alldeaths' in df.columns:
    df['log_deaths'] = np.log1p(df['alldeaths'])

# Drop rows with missing key variables
base_vars = ['log_deaths', 'masfem', 'wind', 'min', 'category', 'year']
model_df = df.dropna(subset=base_vars).copy()

# Model 1: log deaths ~ masfem (bivariate)
model1 = smf.ols('log_deaths ~ masfem', data=model_df).fit(cov_type='HC3')

# Model 2: add intensity controls
model2 = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=model_df).fit(cov_type='HC3')

# Model 3: binary gender indicator instead of masfem
model3 = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=model_df.dropna(subset=['gender_mf'])).fit(cov_type='HC3')

# Spearman correlation between masfem and deaths
spearman = model_df[['masfem', 'alldeaths']].corr(method='spearman').iloc[0, 1]

# Extract key stats
results = {
    'n': int(model_df.shape[0]),
    'model1': {
        'coef_masfem': model1.params.get('masfem', np.nan),
        'p_masfem': model1.pvalues.get('masfem', np.nan),
        'r2': model1.rsquared,
    },
    'model2': {
        'coef_masfem': model2.params.get('masfem', np.nan),
        'p_masfem': model2.pvalues.get('masfem', np.nan),
        'r2': model2.rsquared,
    },
    'model3': {
        'coef_gender_mf': model3.params.get('gender_mf', np.nan),
        'p_gender_mf': model3.pvalues.get('gender_mf', np.nan),
        'r2': model3.rsquared,
    },
    'spearman_masfem_alldeaths': spearman,
}

print(json.dumps(results, indent=2, default=float))
