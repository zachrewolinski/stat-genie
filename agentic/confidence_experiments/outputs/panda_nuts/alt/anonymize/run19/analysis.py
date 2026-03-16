import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename columns for clarity
col_map = {
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'help'
}
df = df.rename(columns=col_map)

# Clean categories
# Standardize case
for col in ['sex', 'hammer', 'help']:
    df[col] = df[col].astype(str).str.strip()

# Compute efficiency: nuts opened per minute
# Add small epsilon to avoid division by zero if any
epsilon = 1e-9
df['efficiency_per_min'] = df['nuts_opened'] / ((df['duration_sec'] + epsilon) / 60.0)

# Basic sanity
print('Rows:', len(df))
print(df[['age','sex','help','nuts_opened','duration_sec','efficiency_per_min']].head())

# Regression model: efficiency ~ age + sex + help
# Treat sex and help as categorical
model = smf.ols('efficiency_per_min ~ age + C(sex) + C(help)', data=df).fit()
print('\nOLS summary (efficiency per min):')
print(model.summary())

# Also check if age, sex, help jointly significant (ANOVA)
anova = sm.stats.anova_lm(model, typ=2)
print('\nANOVA (type II):')
print(anova)

# Compute effect sizes for categorical comparisons
# group means
print('\nGroup means (efficiency per min) by sex:')
print(df.groupby('sex')['efficiency_per_min'].describe())
print('\nGroup means (efficiency per min) by help:')
print(df.groupby('help')['efficiency_per_min'].describe())

# Simple correlations
print('\nCorrelation age vs efficiency:')
print(df['age'].corr(df['efficiency_per_min']))

# Alternative model: log efficiency to reduce skew
# Avoid log of zero by adding small constant
min_eff = df['efficiency_per_min'].min()
add = 0.001 if min_eff <= 0 else 0

df['log_eff'] = np.log(df['efficiency_per_min'] + add)
model_log = smf.ols('log_eff ~ age + C(sex) + C(help)', data=df).fit()
print('\nOLS summary (log efficiency):')
print(model_log.summary())

anova_log = sm.stats.anova_lm(model_log, typ=2)
print('\nANOVA (log efficiency, type II):')
print(anova_log)

# Save key stats for later
out = {
    'model_params': model.params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_r2': model.rsquared,
    'anova': anova['PR(>F)'].to_dict(),
    'model_log_params': model_log.params.to_dict(),
    'model_log_pvalues': model_log.pvalues.to_dict(),
    'model_log_r2': model_log.rsquared,
    'anova_log': anova_log['PR(>F)'].to_dict(),
}

import json
with open('analysis_results.json','w') as f:
    json.dump(out, f, indent=2)

