import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data

df = pd.read_csv('panda_nuts.csv')

# Rename for clarity

df = df.rename(columns={
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'help'
})

# Efficiency: nuts opened per second

df['efficiency'] = df['nuts_opened'] / df['duration_sec']

# Basic checks

print('Rows:', len(df))
print('Efficiency summary (nuts/sec):')
print(df['efficiency'].describe())

# Encode categorical variables

# Ensure categories and consistent baseline

df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# OLS with robust SE
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')
print('\nOLS (robust HC3)')
print(model.summary())

# Also run log(1+efficiency) as sensitivity (handle zeros)

log_model = smf.ols('np.log1p(efficiency) ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')
print('\nOLS on log1p(efficiency) (robust HC3)')
print(log_model.summary())

# Group means for interpretation

print('\nGroup means (efficiency) by sex:')
print(df.groupby('sex')['efficiency'].mean())
print('\nGroup means (efficiency) by help:')
print(df.groupby('help')['efficiency'].mean())

# Correlation with age

print('\nCorrelation age vs efficiency:')
print(df[['age','efficiency']].corr())

# Partial F-test for joint significance of age+sex+help (compare to intercept only)

model_null = smf.ols('efficiency ~ 1', data=df).fit()

# Use anova for model comparison

anova = sm.stats.anova_lm(model_null, smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit())
print('\nANOVA (null vs full):')
print(anova)
