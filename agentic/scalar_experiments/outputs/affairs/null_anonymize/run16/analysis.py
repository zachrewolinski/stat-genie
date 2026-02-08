import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# feature2: frequency of affairs past year
# feature6: children in marriage (yes/no)

# Clean if needed

df = df.copy()

# Group stats

groups = df.groupby('feature6')
summary = groups['feature2'].agg(['count','mean','median','std'])
print('Summary feature2 by children:\n', summary, '\n')

# Any affair binary

df['any_affair'] = df['feature2'] > 0
prop = groups['any_affair'].mean()
print('Proportion any affair by children:\n', prop, '\n')

# Welch t-test for feature2

yes = df[df['feature6']=='yes']['feature2']
no = df[df['feature6']=='no']['feature2']

# some datasets may have string case
if yes.empty or no.empty:
    yes = df[df['feature6'].str.lower()=='yes']['feature2']
    no = df[df['feature6'].str.lower()=='no']['feature2']


t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False)
print('Welch t-test feature2 yes vs no:', t_stat, p_val)

# Cohen's d
n1, n2 = len(yes), len(no)
# pooled std (unbiased)
var1, var2 = yes.var(ddof=1), no.var(ddof=1)
pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2)/(n1+n2-2))
cohen_d = (yes.mean() - no.mean()) / pooled
print('Cohen d (yes-no):', cohen_d)

# Chi-square for any affair

cont = pd.crosstab(df['feature6'], df['any_affair'])
print('Contingency table:\n', cont, '\n')

chi2, chi_p, dof, expected = stats.chi2_contingency(cont)
print('Chi-square:', chi2, 'p=', chi_p)

# Odds ratio for any affair (yes vs no)
# a= yes & affair, b= yes & no affair, c= no & affair, d= no & no affair

if 'yes' in cont.index and 'no' in cont.index:
    a = cont.loc['yes', True]
    b = cont.loc['yes', False]
    c = cont.loc['no', True]
    d = cont.loc['no', False]
    # Haldane-Anscombe correction if any zero
    if min(a,b,c,d) == 0:
        a,b,c,d = a+0.5, b+0.5, c+0.5, d+0.5
    odds_ratio = (a/b) / (c/d)
    print('Odds ratio (yes/no) for any affair:', odds_ratio)
else:
    print('Missing yes/no categories for odds ratio')
