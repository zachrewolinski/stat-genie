import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Map variables using metadata (column names appear shuffled)
# 'religiousness' column is described as children yes/no
children = df['religiousness']
# 'age' column is described as affairs frequency
affairs = df['age']

# Group stats
summary = affairs.groupby(children).agg(['count','mean','std','median'])

# Welch t-test
yes = affairs[children == 'yes']
no = affairs[children == 'no']

t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False)

# Mann-Whitney U (two-sided)
try:
    u_stat, p_u = stats.mannwhitneyu(yes, no, alternative='two-sided')
except ValueError:
    u_stat, p_u = np.nan, np.nan

# Effect size (Cohen's d with pooled SD)
pooled_sd = np.sqrt(((len(yes)-1)*yes.var(ddof=1) + (len(no)-1)*no.var(ddof=1)) / (len(yes)+len(no)-2))
cohens_d = (yes.mean() - no.mean()) / pooled_sd

# Difference in means and 95% CI (Welch)
mean_diff = yes.mean() - no.mean()
# Welch-Satterthwaite df
se = np.sqrt(yes.var(ddof=1)/len(yes) + no.var(ddof=1)/len(no))
df_welch = (yes.var(ddof=1)/len(yes) + no.var(ddof=1)/len(no))**2 / ((yes.var(ddof=1)/len(yes))**2/(len(yes)-1) + (no.var(ddof=1)/len(no))**2/(len(no)-1))
ci_low, ci_high = stats.t.interval(0.95, df_welch, loc=mean_diff, scale=se)

print('Group summary')
print(summary)
print('\nWelch t-test: t=%.4f p=%.6f' % (t_stat, p_val))
print('Mann-Whitney U: U=%.2f p=%.6f' % (u_stat, p_u))
print('Mean diff (yes - no)=%.4f, 95%% CI [%.4f, %.4f]' % (mean_diff, ci_low, ci_high))
print('Cohen d=%.4f' % cohens_d)
