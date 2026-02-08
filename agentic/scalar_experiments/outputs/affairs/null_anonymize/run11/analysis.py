import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('affairs.csv')

# Map children yes/no
children = df['feature6'].str.lower()

affairs = df['feature2']
any_affair = (affairs > 0).astype(int)

summary = df.groupby(children).agg(
    n=('feature2','size'),
    mean_affairs=('feature2','mean'),
    median_affairs=('feature2','median'),
    prop_any_affair=('feature2', lambda s: (s>0).mean())
)

# Difference yes - no (children yes minus no)
if set(summary.index) >= {'yes','no'}:
    diff_mean = summary.loc['yes','mean_affairs'] - summary.loc['no','mean_affairs']
    diff_prop = summary.loc['yes','prop_any_affair'] - summary.loc['no','prop_any_affair']
else:
    diff_mean = np.nan
    diff_prop = np.nan

# t-test for mean affairs (Welch)
children_yes = affairs[children=='yes']
children_no = affairs[children=='no']

ttest = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')

# test for proportion difference (z test)
count_yes = (children_yes>0).sum()
count_no = (children_no>0).sum()

n_yes = len(children_yes)
n_no = len(children_no)

p_pool = (count_yes + count_no) / (n_yes + n_no)
se = np.sqrt(p_pool*(1-p_pool)*(1/n_yes + 1/n_no))
if se>0:
    z = (count_yes/n_yes - count_no/n_no) / se
    p_value = 2*(1-stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_value = np.nan

print('summary')
print(summary)
print('\nDiff mean (yes-no):', diff_mean)
print('Diff prop any (yes-no):', diff_prop)
print('\nWelch t-test:', ttest)
print('\nProp test: z', z, 'p', p_value)
