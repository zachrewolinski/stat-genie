import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Variables
# feature4: femininity index (1=masculine, 11=feminine)
# feature6: binary female (1) / male (0)
# feature8: deaths
# Controls for storm intensity: category, wind speed, minimum pressure

# Create log deaths
# Add 1 to handle zero deaths

df = df.copy()
df['log_deaths'] = np.log1p(df['feature8'])

# Simple correlations
corr_masfem = df['feature4'].corr(df['log_deaths'])
corr_masfem_raw = df['feature4'].corr(df['feature8'])

# Group comparison for binary gender
male = df[df['feature6'] == 0]['feature8']
female = df[df['feature6'] == 1]['feature8']

# Use Mann-Whitney (non-parametric) and t-test on log deaths
u_stat, u_p = stats.mannwhitneyu(male, female, alternative='two-sided')
t_stat, t_p = stats.ttest_ind(np.log1p(male), np.log1p(female), equal_var=False)

# Regression models
# Model 1: log deaths ~ femininity index + controls
model1 = smf.ols('log_deaths ~ feature4 + feature7 + feature13 + feature5 + feature2', data=df).fit(cov_type='HC3')

# Model 2: log deaths ~ binary female + controls
model2 = smf.ols('log_deaths ~ feature6 + feature7 + feature13 + feature5 + feature2', data=df).fit(cov_type='HC3')

# Model 3: log deaths ~ femininity index only
model3 = smf.ols('log_deaths ~ feature4', data=df).fit(cov_type='HC3')

# Collect key results
results = {
    'n': len(df),
    'corr_masfem_log': corr_masfem,
    'corr_masfem_raw': corr_masfem_raw,
    'male_mean_deaths': male.mean(),
    'female_mean_deaths': female.mean(),
    'male_median_deaths': male.median(),
    'female_median_deaths': female.median(),
    'mannwhitney_p': u_p,
    'ttest_log_p': t_p,
    'model1_coef': model1.params['feature4'],
    'model1_p': model1.pvalues['feature4'],
    'model1_ci': model1.conf_int().loc['feature4'].tolist(),
    'model2_coef': model2.params['feature6'],
    'model2_p': model2.pvalues['feature6'],
    'model2_ci': model2.conf_int().loc['feature6'].tolist(),
    'model3_coef': model3.params['feature4'],
    'model3_p': model3.pvalues['feature4'],
    'model3_ci': model3.conf_int().loc['feature4'].tolist(),
}

print(results)
