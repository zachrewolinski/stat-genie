import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('affairs.csv')

# Basic counts
print('Rows:', len(df))
print(df['children'].value_counts())

# Mean affairs by children
means = df.groupby('children')['affairs'].agg(['mean','median','count'])
print('\nMean/median affairs by children')
print(means)

# Proportion with any affair
any_affair = df.assign(any_affair=df['affairs']>0)
prop_any = any_affair.groupby('children')['any_affair'].mean()
print('\nProportion any affair by children')
print(prop_any)

# t-test on affairs (unequal variance)
children_yes = df.loc[df['children']=='yes','affairs']
children_no = df.loc[df['children']=='no','affairs']

ttest = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')
print('\nT-test affairs yes vs no (Welch)')
print(ttest)

# T-test on any_affair (proportion) via z-test
success = any_affair.groupby('children')['any_affair'].sum()
count = any_affair.groupby('children')['any_affair'].count()
# order: no, yes
succ_no, succ_yes = success['no'], success['yes']
count_no, count_yes = count['no'], count['yes']
prop_no = succ_no / count_no
prop_yes = succ_yes / count_yes
p_pool = (succ_no + succ_yes) / (count_no + count_yes)
se = np.sqrt(p_pool * (1 - p_pool) * (1/count_no + 1/count_yes))
if se > 0:
    z = (prop_yes - prop_no) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p = np.nan
print('\nTwo-proportion z-test (any affair)')
print('prop_no', prop_no, 'prop_yes', prop_yes, 'z', z, 'p', p)

# OLS with controls on log(affairs+1)
# Use categorical for gender, children
formula = 'np.log1p(affairs) ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')
print('\nOLS log1p(affairs) with controls')
print(model.summary().tables[1])

# Logistic regression for any affair
any_affair = df.assign(any_affair=(df['affairs']>0).astype(int))
logit_formula = 'any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating'
logit_model = smf.logit(logit_formula, data=any_affair).fit(disp=0)
print('\nLogit any affair with controls')
print(logit_model.summary().tables[1])

# Poisson regression for counts
poisson_model = smf.glm('affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
                        data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
print('\nPoisson affairs with controls')
print(poisson_model.summary().tables[1])
