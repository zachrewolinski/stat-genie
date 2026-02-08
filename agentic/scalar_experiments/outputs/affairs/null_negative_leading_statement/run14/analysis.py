import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


df = pd.read_csv('affairs.csv')

children_yes = df[df['children'] == 'yes']
children_no = df[df['children'] == 'no']

summary = {
    'n_total': len(df),
    'n_yes': len(children_yes),
    'n_no': len(children_no),
    'mean_yes': children_yes['affairs'].mean(),
    'mean_no': children_no['affairs'].mean(),
    'median_yes': children_yes['affairs'].median(),
    'median_no': children_no['affairs'].median(),
    'prop_affairs_yes': (children_yes['affairs'] > 0).mean(),
    'prop_affairs_no': (children_no['affairs'] > 0).mean(),
}

welch = stats.ttest_ind(children_yes['affairs'], children_no['affairs'], equal_var=False)

mw = stats.mannwhitneyu(children_yes['affairs'], children_no['affairs'], alternative='two-sided')

mean_yes = summary['mean_yes']
mean_no = summary['mean_no']
var_yes = children_yes['affairs'].var(ddof=1)
var_no = children_no['affairs'].var(ddof=1)

n1 = len(children_yes)
n2 = len(children_no)
pooled_sd = np.sqrt(((n1-1)*var_yes + (n2-1)*var_no) / (n1+n2-2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

count_yes = (children_yes['affairs'] > 0).sum()
count_no = (children_no['affairs'] > 0).sum()
prop_test = proportions_ztest([count_yes, count_no], [n1, n2])

print('SUMMARY')
for k, v in summary.items():
    print(f'{k}: {v}')

print('\nTESTS')
print('welch_ttest_stat:', welch.statistic, 'p:', welch.pvalue)
print('mannwhitney_u:', mw.statistic, 'p:', mw.pvalue)
print('cohens_d (yes - no):', cohens_d)
print('prop_ztest_stat:', prop_test[0], 'p:', prop_test[1])
