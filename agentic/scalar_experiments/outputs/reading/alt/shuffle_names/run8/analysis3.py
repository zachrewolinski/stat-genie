import pandas as pd
import numpy as np
from scipy import stats

path='reading.csv'
df=pd.read_csv(path)

# compute wpm
wpm = df['num_words'] / (df['adjusted_running_time']/60000)
df = df.assign(wpm=wpm)

# dyslexia
_df = df[df['dyslexia_bin']==1].copy()

# reader view indicator
rv = 'language'

rv1 = _df[_df[rv]==1]['wpm']
rv0 = _df[_df[rv]==0]['wpm']

print('wpm means', rv1.mean(), rv0.mean())
print('welch', stats.ttest_ind(rv1, rv0, equal_var=False))

# participant id
id_col='speed'
by_id = _df.groupby(id_col)[rv].nunique()
ids_both = by_id[by_id==2].index
paired = _df[_df[id_col].isin(ids_both)]
pivot = paired.pivot_table(index=id_col, columns=rv, values='wpm', aggfunc='mean')
print('paired means', pivot[1].mean(), pivot[0].mean())
print('paired t', stats.ttest_rel(pivot[1], pivot[0]))

