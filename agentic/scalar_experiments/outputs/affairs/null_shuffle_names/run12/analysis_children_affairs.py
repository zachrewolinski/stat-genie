import pandas as pd
from scipy import stats

_df = pd.read_csv('affairs.csv')
# Map: affairs count is in column 'age'; children indicator in 'religiousness'

affairs = _df['age']
children = _df['religiousness']

# binary: yes/no
# compute summary
summary = _df.groupby(children).agg(
    n=('age','size'),
    mean_affairs=('age','mean'),
    median_affairs=('age','median'),
    prop_any_affair=('age', lambda x: (x>0).mean())
)
print(summary)

# t-test (Welch) on affairs counts
no_affairs = affairs[children=='no']
yes_affairs = affairs[children=='yes']

t_stat, p_val = stats.ttest_ind(yes_affairs, no_affairs, equal_var=False)
print('Welch t-test: t', t_stat, 'p', p_val)

# Mann-Whitney U
u_stat, p_u = stats.mannwhitneyu(yes_affairs, no_affairs, alternative='two-sided')
print('Mann-Whitney U: U', u_stat, 'p', p_u)

# difference in means and standardized effect size (Cohen d using pooled sd?)
mean_diff = yes_affairs.mean() - no_affairs.mean()
# use pooled sd for d
n1, n2 = len(yes_affairs), len(no_affairs)
var1, var2 = yes_affairs.var(ddof=1), no_affairs.var(ddof=1)
pooled_sd = (( (n1-1)*var1 + (n2-1)*var2 ) / (n1+n2-2))**0.5
cohen_d = mean_diff / pooled_sd
print('mean_diff', mean_diff, 'cohen_d', cohen_d)

