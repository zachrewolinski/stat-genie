import pandas as pd
import scipy.stats as stats

df = pd.read_csv('affairs.csv')

affairs = df['age']
children = df['religiousness'].map({'yes':1,'no':0})

x = affairs[children==1]
y = affairs[children==0]

# Welch t-test
res = stats.ttest_ind(x, y, equal_var=False)

# Mann-Whitney U
mw = stats.mannwhitneyu(x, y, alternative='two-sided')

print('welch_t', res.statistic, res.pvalue)
print('mw', mw.statistic, mw.pvalue)
