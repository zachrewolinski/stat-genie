import pandas as pd
import numpy as np
from scipy import stats

path='reading.csv'
df=pd.read_csv(path)

# identify columns
reader_view = 'language'  # 0/1
speed_col = 'running_time'  # reading speed
id_col = 'speed'  # participant id

# dyslexia indicator
# prefer dyslexia_bin if binary
print('dyslexia_bin unique', df['dyslexia_bin'].unique())

# filter dyslexia
df_dys = df[df['dyslexia_bin'] == 1].copy()
print('dyslexia rows', len(df_dys))

# group sizes
print('reader_view counts', df_dys[reader_view].value_counts())

# independent t-test
rv1 = df_dys[df_dys[reader_view]==1][speed_col].dropna()
rv0 = df_dys[df_dys[reader_view]==0][speed_col].dropna()

print('means', rv1.mean(), rv0.mean())
print('stds', rv1.std(), rv0.std())

# Welch t-test
wt = stats.ttest_ind(rv1, rv0, equal_var=False)
print('welch', wt)

# effect size Cohen d
n1, n0 = len(rv1), len(rv0)
# pooled std (for Cohen's d)
s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2)/(n1+n0-2))
d = (rv1.mean() - rv0.mean())/sp
print('cohen d', d)

# check within-subject availability: participants with both conditions
by_id = df_dys.groupby(id_col)[reader_view].nunique()
print('participants total', len(by_id))
print('participants with both conditions', (by_id==2).sum())

# paired analysis for those with both conditions
ids_both = by_id[by_id==2].index
paired = df_dys[df_dys[id_col].isin(ids_both)].copy()
# compute participant-level mean speed by condition
pivot = paired.pivot_table(index=id_col, columns=reader_view, values=speed_col, aggfunc='mean')
# ensure columns 0 and 1
if 0 in pivot.columns and 1 in pivot.columns:
    diff = pivot[1] - pivot[0]
    print('paired mean diff', diff.mean())
    print('paired t-test', stats.ttest_rel(pivot[1], pivot[0]))
else:
    print('paired analysis not possible')

