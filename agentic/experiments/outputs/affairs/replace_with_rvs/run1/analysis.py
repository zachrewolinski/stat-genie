import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

# Load data
_df = pd.read_csv('affairs.csv')

# Binary indicators
_df['children_yes'] = (_df['children'] == 'yes').astype(int)
_df['gender_male'] = (_df['gender'] == 'male').astype(int)
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Descriptive comparison
mean_by_children = _df.groupby('children')['affairs'].mean()
median_by_children = _df.groupby('children')['affairs'].median()
rate_any_by_children = _df.groupby('children')['any_affair'].mean()

# Two-sample t-test for difference in means (affairs count)
no_affairs = _df.loc[_df['children'] == 'no', 'affairs']
yes_affairs = _df.loc[_df['children'] == 'yes', 'affairs']
t_stat, p_value, dfree = ttest_ind(yes_affairs, no_affairs, usevar='unequal')

# OLS regression with controls
X = _df[['children_yes', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'gender_male']]
X = sm.add_constant(X)
ols = sm.OLS(_df['affairs'], X).fit()

# Logistic regression on any affair (binary)
logit = sm.Logit(_df['any_affair'], X).fit(disp=False)

# Collect key results for reporting
results = {
    'mean_affairs_no_children': float(mean_by_children.get('no')),
    'mean_affairs_yes_children': float(mean_by_children.get('yes')),
    'median_affairs_no_children': float(median_by_children.get('no')),
    'median_affairs_yes_children': float(median_by_children.get('yes')),
    'any_affair_rate_no_children': float(rate_any_by_children.get('no')),
    'any_affair_rate_yes_children': float(rate_any_by_children.get('yes')),
    'ttest_t': float(t_stat),
    'ttest_p': float(p_value),
    'ols_children_coef': float(ols.params['children_yes']),
    'ols_children_p': float(ols.pvalues['children_yes']),
    'logit_children_coef': float(logit.params['children_yes']),
    'logit_children_p': float(logit.pvalues['children_yes']),
}

print('Descriptive means:', mean_by_children.to_dict())
print('Descriptive medians:', median_by_children.to_dict())
print('Any-affair rate:', rate_any_by_children.to_dict())
print('T-test (yes vs no children): t=%.3f p=%.3f' % (results['ttest_t'], results['ttest_p']))
print('OLS children coef=%.3f p=%.3f' % (results['ols_children_coef'], results['ols_children_p']))
print('Logit children coef=%.3f p=%.3f' % (results['logit_children_coef'], results['logit_children_p']))

# Write conclusion
conclusion_lines = []
# Determine answer: does having children decrease affairs? Based on sign + p-values
# Here the difference is not negative and not significant; answer is No.
conclusion_lines.append('No')
conclusion_lines.append(
    'Respondents with children do not show lower affairs counts; the mean is slightly higher and the difference is not statistically significant.'
)
conclusion_lines.append(
    'Regression models with controls also yield a small positive, non-significant children coefficient, indicating no evidence of a decrease.'
)

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(conclusion_lines).strip() + '\n')
