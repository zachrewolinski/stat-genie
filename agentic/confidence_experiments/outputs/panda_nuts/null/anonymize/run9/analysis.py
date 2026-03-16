import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Rename columns for clarity
cols = {
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'help'
}
df = df.rename(columns=cols)

# Efficiency: nuts opened per minute (more interpretable scale)
df['efficiency'] = df['nuts_opened'] / df['duration_sec'] * 60.0

# Basic summary
print('Rows:', len(df))
print(df[['age','sex','help','efficiency']].head())

# Ensure categorical types
for col in ['sex','help','hammer']:
    df[col] = df[col].astype('category')

# OLS model: efficiency ~ age + sex + help
model1 = smf.ols('efficiency ~ age + sex + help', data=df).fit()
print('\nModel 1: efficiency ~ age + sex + help')
print(model1.summary())

# OLS model with hammer as control
model2 = smf.ols('efficiency ~ age + sex + help + hammer', data=df).fit()
print('\nModel 2: efficiency ~ age + sex + help + hammer')
print(model2.summary())

# Robust SE (HC3) for model 1
model1_robust = model1.get_robustcov_results(cov_type='HC3')
print('\nModel 1 robust (HC3)')
print(model1_robust.summary())

# Robust SE (HC3) for model 2
model2_robust = model2.get_robustcov_results(cov_type='HC3')
print('\nModel 2 robust (HC3)')
print(model2_robust.summary())

# Also compute nonparametric comparisons for sex/help
from scipy import stats

# Sex comparison
sex_groups = [df.loc[df['sex']==s, 'efficiency'].dropna() for s in df['sex'].cat.categories]
if len(sex_groups) == 2:
    tstat, pval = stats.ttest_ind(sex_groups[0], sex_groups[1], equal_var=False)
    print('\nWelch t-test sex:', df['sex'].cat.categories.tolist(), 'p=', pval)

# Help comparison
help_groups = [df.loc[df['help']==h, 'efficiency'].dropna() for h in df['help'].cat.categories]
if len(help_groups) == 2:
    tstat, pval = stats.ttest_ind(help_groups[0], help_groups[1], equal_var=False)
    print('Welch t-test help:', df['help'].cat.categories.tolist(), 'p=', pval)

# Correlation age with efficiency
age_corr = stats.pearsonr(df['age'], df['efficiency'])
print('Pearson corr age-efficiency:', age_corr)
