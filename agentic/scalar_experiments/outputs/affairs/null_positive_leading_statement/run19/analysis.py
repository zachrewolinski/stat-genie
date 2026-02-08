import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Basic cleaning
# Ensure children is categorical yes/no
_df['children'] = _df['children'].astype('category')

# Create binary outcome: any affair >0
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Group summaries
summary = _df.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    median_affairs=('affairs','median'),
    any_affair_rate=('any_affair','mean')
)

# Difference in means
mean_no = summary.loc['no','mean_affairs']
mean_yes = summary.loc['yes','mean_affairs']
rate_no = summary.loc['no','any_affair_rate']
rate_yes = summary.loc['yes','any_affair_rate']

# Simple t-test (Welch) for affairs
from scipy import stats

aff_no = _df.loc[_df['children']=='no','affairs']
aff_yes = _df.loc[_df['children']=='yes','affairs']

ttest = stats.ttest_ind(aff_yes, aff_no, equal_var=False)

# Logistic regression for any affair with covariates
# children as binary: yes=1
_df['children_yes'] = (_df['children']=='yes').astype(int)

# Add common covariates in Fair data
covars = ['children_yes','gender','age','yearsmarried','religiousness','education','occupation','rating']

# Logistic model
logit_model = smf.logit('any_affair ~ children_yes + gender + age + yearsmarried + religiousness + education + occupation + rating', data=_df).fit(disp=False)

# OLS on affairs (censored but use as linear)
ols_model = smf.ols('affairs ~ children_yes + gender + age + yearsmarried + religiousness + education + occupation + rating', data=_df).fit()

# Collect key stats
result = {
    'summary': summary,
    'mean_diff': mean_yes - mean_no,
    'rate_diff': rate_yes - rate_no,
    'ttest_stat': ttest.statistic,
    'ttest_p': ttest.pvalue,
    'logit_coef_children': logit_model.params['children_yes'],
    'logit_p_children': logit_model.pvalues['children_yes'],
    'ols_coef_children': ols_model.params['children_yes'],
    'ols_p_children': ols_model.pvalues['children_yes'],
}

print('GROUP SUMMARY')
print(summary)
print('\nMEAN DIFFERENCE (yes - no):', result['mean_diff'])
print('ANY AFFAIR RATE DIFF (yes - no):', result['rate_diff'])
print('TTEST affairs (yes vs no): stat=', result['ttest_stat'], 'p=', result['ttest_p'])
print('\nLOGIT children_yes coef:', result['logit_coef_children'], 'p=', result['logit_p_children'])
print('OLS children_yes coef:', result['ols_coef_children'], 'p=', result['ols_p_children'])
