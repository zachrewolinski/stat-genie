import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns
# feature1: ID, feature2: age, feature3: sex, feature4: hammer type, feature5: nuts opened, feature6: duration seconds, feature7: help yes/no

df = df.copy()

# Clean categorical fields
# Ensure consistent categories
for col in ['feature3', 'feature4', 'feature7']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Compute efficiency: nuts opened per second
# Avoid division by zero (though min duration 2.5 per metadata)

df['efficiency'] = df['feature5'] / df['feature6']

# Basic summary
summary = df[['efficiency','feature2','feature3','feature7']].describe(include='all')
print('Summary:\n', summary)

# Model: efficiency ~ age + sex + help
# Use OLS with categorical variables
model = smf.ols('efficiency ~ feature2 + C(feature3) + C(feature7)', data=df).fit()
print('\nOLS Summary:\n', model.summary())

# ANOVA for categorical influence
anova = anova_lm(model, typ=2)
print('\nANOVA (type II):\n', anova)

# Group means for sex and help
means_sex = df.groupby('feature3')['efficiency'].mean()
means_help = df.groupby('feature7')['efficiency'].mean()
print('\nMeans by sex:\n', means_sex)
print('\nMeans by help:\n', means_help)

# Correlation between age and efficiency (Pearson)
from scipy import stats
r, p = stats.pearsonr(df['feature2'], df['efficiency'])
print('\nPearson r age-efficiency:', r, 'p=', p)

# Also consider log efficiency if skewed? Provide alternative model
# Add small epsilon to avoid log(0)
if (df['efficiency'] <= 0).any():
    eps = 1e-6
else:
    eps = 0

df['log_efficiency'] = np.log(df['efficiency'] + eps)
log_model = smf.ols('log_efficiency ~ feature2 + C(feature3) + C(feature7)', data=df).fit()
print('\nLog OLS Summary:\n', log_model.summary())

anova_log = anova_lm(log_model, typ=2)
print('\nANOVA log (type II):\n', anova_log)
