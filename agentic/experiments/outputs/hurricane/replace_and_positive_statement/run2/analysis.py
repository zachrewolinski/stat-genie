import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic checks
print('Rows:', len(_df))
print('Columns:', list(_df.columns))

# Outcome: log(1 + deaths) to reduce skew
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Simple correlations
corr_raw = _df['masfem'].corr(_df['alldeaths'])
corr_log = _df['masfem'].corr(_df['log_deaths'])
print('Correlation masfem vs deaths:', corr_raw)
print('Correlation masfem vs log deaths:', corr_log)

# Regression controlling for storm intensity and era
features = ['masfem', 'wind', 'min', 'category', 'ndam15', 'elapsedyrs']
X = sm.add_constant(_df[features])
model = sm.OLS(_df['log_deaths'], X).fit()
print(model.summary())

# Also check binary gender for robustness
features_bin = ['gender_mf', 'wind', 'min', 'category', 'elapsedyrs']
X2 = sm.add_constant(_df[features_bin])
model2 = sm.OLS(_df['log_deaths'], X2).fit()
print(model2.summary())

# Save key results for quick reference
results = {
    'corr_raw': float(corr_raw),
    'corr_log': float(corr_log),
    'masfem_coef': float(model.params['masfem']),
    'masfem_pval': float(model.pvalues['masfem']),
    'gender_mf_coef': float(model2.params['gender_mf']),
    'gender_mf_pval': float(model2.pvalues['gender_mf']),
}
print('Key results:', results)
