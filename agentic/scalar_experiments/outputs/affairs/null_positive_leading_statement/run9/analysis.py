import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic preprocessing
# children is 'yes'/'no'

df['children_yes'] = (df['children'].str.lower() == 'yes').astype(int)

# Outcome variants

df['any_affair'] = (df['affairs'] > 0).astype(int)

df['log1p_affairs'] = np.log1p(df['affairs'])

# Group stats

group = df.groupby('children')['affairs']

stats_table = group.agg(['count','mean','median','std'])

# Proportion any affair

group_any = df.groupby('children')['any_affair'].mean()

# T-test on log1p affairs

log_yes = df.loc[df['children_yes']==1,'log1p_affairs']
log_no = df.loc[df['children_yes']==0,'log1p_affairs']

ttest = stats.ttest_ind(log_yes, log_no, equal_var=False)

# Mann-Whitney U on raw affairs

mw = stats.mannwhitneyu(df.loc[df['children_yes']==1,'affairs'], df.loc[df['children_yes']==0,'affairs'], alternative='two-sided')

# Chi-square for any affair

cont = pd.crosstab(df['children'], df['any_affair'])
chi2 = stats.chi2_contingency(cont)

# Regression: OLS on log1p_affairs with controls

# Controls: gender, age, yearsmarried, religiousness, education, occupation, rating

ols = smf.ols('log1p_affairs ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit(cov_type='HC3')

# Logistic regression for any_affair

logit = smf.logit('any_affair ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit(disp=False)

# Poisson regression for counts

pois = smf.glm('affairs ~ children_yes + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df, family=sm.families.Poisson()).fit()

print('Group stats (affairs):')
print(stats_table)
print('\nProportion any affair:')
print(group_any)
print('\nT-test log1p(affairs) children yes vs no:')
print(ttest)
print('\nMann-Whitney U affairs:')
print(mw)
print('\nChi-square any affair:')
print(chi2)

print('\nOLS log1p_affairs coefficient for children_yes:')
print(ols.params['children_yes'], ols.pvalues['children_yes'])

print('\nLogit any_affair coefficient for children_yes:')
print(logit.params['children_yes'], logit.pvalues['children_yes'])

print('\nPoisson affairs coefficient for children_yes:')
print(pois.params['children_yes'], pois.pvalues['children_yes'])
