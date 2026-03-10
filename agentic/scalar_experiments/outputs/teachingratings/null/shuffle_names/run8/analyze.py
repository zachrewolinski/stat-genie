import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic sanity
n = len(df)

# Ensure types
# allstudents is numeric
# beauty is numeric

# Simple correlation
corr = df['beauty'].corr(df['allstudents'])

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit(cov_type='HC3')

# Controlled OLS with plausible covariates
# Treat categorical variables as categories
categoricals = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
for col in categoricals:
    df[col] = df[col].astype('category')

# Use formula with categorical covariates and numeric controls
formula = 'allstudents ~ beauty + age + C(eval) + C(tenure) + C(prof) + C(native) + C(gender) + C(credits) + rownames + minority + students'
model_controls = smf.ols(formula, data=df).fit(cov_type='HC3')

# Extract key stats
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']

controls_coef = model_controls.params['beauty']
controls_p = model_controls.pvalues['beauty']

# Effect size per 1 SD of beauty
beauty_sd = df['beauty'].std()
allstudents_sd = df['allstudents'].std()

simple_effect_sd = simple_coef * beauty_sd
controls_effect_sd = controls_coef * beauty_sd

print('N', n)
print('beauty mean', df['beauty'].mean(), 'sd', beauty_sd)
print('allstudents mean', df['allstudents'].mean(), 'sd', allstudents_sd)
print('corr', corr)
print('simple coef', simple_coef, 'p', simple_p)
print('controls coef', controls_coef, 'p', controls_p)
print('simple effect (1 SD beauty) on allstudents', simple_effect_sd)
print('controls effect (1 SD beauty) on allstudents', controls_effect_sd)

# Also compute R2 for models
print('simple R2', model_simple.rsquared)
print('controls R2', model_controls.rsquared)

# Save a tiny summary to a file for convenience
with open('analysis_summary.txt', 'w') as f:
    f.write(f'N {n}\n')
    f.write(f'corr {corr}\n')
    f.write(f'simple coef {simple_coef} p {simple_p}\n')
    f.write(f'controls coef {controls_coef} p {controls_p}\n')
    f.write(f'simple R2 {model_simple.rsquared}\n')
    f.write(f'controls R2 {model_controls.rsquared}\n')
    f.write(f'beauty sd {beauty_sd}\n')
    f.write(f'allstudents sd {allstudents_sd}\n')
    f.write(f'simple effect sd {simple_effect_sd}\n')
    f.write(f'controls effect sd {controls_effect_sd}\n')
