import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Map children yes/no
df['children'] = df['feature6'].astype(str)

# Outcome: affairs frequency
outcome = df['feature2']

# Basic group stats
summary = df.groupby('children')['feature2'].agg(['count','mean','median','std'])
print('Group summary:')
print(summary)

# T-test (Welch)
children_yes = df.loc[df['children']=='yes','feature2']
children_no = df.loc[df['children']=='no','feature2']

t_stat, p_val = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')
print('\nWelch t-test (yes vs no):', t_stat, p_val)

# Mann-Whitney U test (nonparametric)
try:
    u_stat, u_p = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')
    print('Mann-Whitney U:', u_stat, u_p)
except Exception as e:
    print('Mann-Whitney failed:', e)

# Effect size: Cohen's d
mean_yes = children_yes.mean()
mean_no = children_no.mean()
std_yes = children_yes.std(ddof=1)
std_no = children_no.std(ddof=1)

n_yes = children_yes.shape[0]
n_no = children_no.shape[0]

# pooled sd for d (unequal sizes)
pooled_sd = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes+n_no-2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd != 0 else np.nan
print('Cohen d (yes-no):', cohen_d)

# Regression: OLS with controls
# Use feature2 as dependent, include children and controls
# controls: feature3 gender, feature4 age, feature5 years married, feature7 relig, feature8 edu, feature9 occupation, feature10 marriage rating

# Encode children yes=1, no=0 for regression

df['children_bin'] = (df['children']=='yes').astype(int)

# We'll also use categorical for gender

formula = 'feature2 ~ children_bin + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
model = smf.ols(formula, data=df).fit()
print('\nOLS regression summary (children_bin coefficient):')
print(model.params['children_bin'], model.pvalues['children_bin'])

# Also do robust standard errors
model_robust = model.get_robustcov_results(cov_type='HC3')
print('OLS robust (HC3) children_bin coef & pvalue:')
print(model_robust.params[model_robust.model.exog_names.index('children_bin')], model_robust.pvalues[model_robust.model.exog_names.index('children_bin')])

# Since feature2 is skewed count-like, also try Poisson regression
# Add small constant? but feature2 can be negative per description? Actually sample min is -8.315, so not count. So Poisson not suitable.

# Instead, try Tobit? Not necessary; we can also do median regression
try:
    quant_model = smf.quantreg(formula, data=df).fit(q=0.5)
    print('Quantile (median) regression children_bin coef & pvalue:')
    print(quant_model.params['children_bin'], quant_model.pvalues['children_bin'])
except Exception as e:
    print('Quantile regression failed:', e)

# Print mean difference
print('\nMean difference (yes-no):', mean_yes-mean_no)
