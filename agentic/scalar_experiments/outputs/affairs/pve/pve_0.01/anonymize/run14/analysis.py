import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

print('Columns:', list(_df.columns))
print('Head:')
print(_df.head())

# Identify columns
# feature2: affairs frequency
# feature6: children yes/no

# Basic cleaning
_df = _df.copy()

# Normalize children column to lower-case strings
_df['children'] = _df['feature6'].astype(str).str.strip().str.lower()

# Outcome variables
_df['affairs'] = _df['feature2']
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Group stats
summary = _df.groupby('children')['affairs'].agg(['count','mean','median','std'])
print('\nAffairs (feature2) by children:')
print(summary)

# Proportion with any affairs
prop = _df.groupby('children')['any_affair'].mean().to_frame('prop_any_affair')
prop['count'] = _df.groupby('children')['any_affair'].size()
print('\nProportion any affair by children:')
print(prop)

# Two-sample t-test (Welch) on numeric affairs
child_yes = _df[_df['children'] == 'yes']['affairs']
child_no = _df[_df['children'] == 'no']['affairs']

# Welch t-test
welch = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy='omit')
print('\nWelch t-test (affairs):', welch)

# Mann-Whitney U test (nonparametric)
# Use two-sided alternative
try:
    mwu = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
    print('Mann-Whitney U:', mwu)
except Exception as e:
    print('Mann-Whitney error:', e)

# Effect size: Cohen's d (Welch)
mean_yes = child_yes.mean()
mean_no = child_no.mean()
std_yes = child_yes.std(ddof=1)
std_no = child_no.std(ddof=1)
# pooled SD for unequal sizes
n_yes = child_yes.shape[0]
n_no = child_no.shape[0]
pooled_sd = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes + n_no - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd
print('Cohen d (yes - no):', cohens_d)

# Proportion test for any_affair
# Use chi-square test of independence
cont_table = pd.crosstab(_df['children'], _df['any_affair'])
print('\nContingency table (children x any_affair):')
print(cont_table)
chi2, p_chi2, dof, expected = stats.chi2_contingency(cont_table)
print('Chi-square test:', (chi2, p_chi2, dof))

# Difference in proportions and odds ratio
prop_yes = child_yes_gt0 = _df[_df['children']=='yes']['any_affair'].mean()
prop_no = child_no_gt0 = _df[_df['children']=='no']['any_affair'].mean()
print('prop_yes:', prop_yes, 'prop_no:', prop_no, 'diff (yes - no):', prop_yes - prop_no)

# Odds ratio from contingency table (add 0.5 to avoid zero if needed)
# table layout: rows yes/no, columns 0/1
# We want odds of any_affair=1 vs 0
if cont_table.shape == (2,2):
    # ensure column order 0,1
    cont_table = cont_table[[0,1]]
    a = cont_table.loc['yes',1]
    b = cont_table.loc['yes',0]
    c = cont_table.loc['no',1]
    d = cont_table.loc['no',0]
    # add 0.5 for safety
    or_val = (a+0.5)*(d+0.5)/((b+0.5)*(c+0.5))
    print('Odds ratio (yes vs no):', or_val)

# Logistic regression for any_affair ~ children + controls (age, years married, relig, rating?)
# Use feature4 (age), feature5 (years married), feature7 (religiousness), feature10 (marriage rating)
_df['children_yes'] = (_df['children'] == 'yes').astype(int)

# Fit logistic regression
model = smf.logit('any_affair ~ children_yes + feature4 + feature5 + feature7 + feature10', data=_df).fit(disp=False)
print('\nLogit model summary (partial):')
print(model.summary2().tables[1])

# OLS on affairs frequency (may be skewed) with same controls
ols_model = smf.ols('affairs ~ children_yes + feature4 + feature5 + feature7 + feature10', data=_df).fit()
print('\nOLS model summary (partial):')
print(ols_model.summary2().tables[1])

