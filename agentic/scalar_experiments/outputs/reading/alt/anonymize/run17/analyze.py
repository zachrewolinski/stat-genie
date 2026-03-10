import pandas as pd
import numpy as np
from scipy import stats

path = "reading.csv"
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all').transpose()[['count','mean','std','min','max']])

# compute derived reading speed (words per minute) using feature7 words and feature5 reading time (ms)
# avoid division by zero
reading_time_ms = df['feature5']
words = df['feature7']
# remove nonpositive times
speed_wpm = words / (reading_time_ms/60000.0)

# attach
_df = df.copy()
_df['speed_wpm'] = speed_wpm

# check correlation with feature20
valid = _df[['speed_wpm','feature20']].replace([np.inf,-np.inf], np.nan).dropna()
if len(valid) > 0:
    corr = stats.pearsonr(valid['speed_wpm'], valid['feature20'])
    print("Correlation speed_wpm vs feature20:", corr)

# inspect distribution of feature20
print("feature20 quantiles", _df['feature20'].quantile([0,0.01,0.05,0.5,0.95,0.99,1]))
print("speed_wpm quantiles", _df['speed_wpm'].replace([np.inf,-np.inf], np.nan).dropna().quantile([0,0.01,0.05,0.5,0.95,0.99,1]))

# define dyslexia group: feature17 ==1 or feature12>0
_df['dyslexia_any'] = (_df['feature17'] == 1) | (_df['feature12'] > 0)

# filter dyslexia participants
sub = _df[_df['dyslexia_any']].copy()
print("dyslexia rows", len(sub))

# ensure reader view column feature3 (1=activated)
# compute per participant mean speed per condition
# use feature1 as participant id
agg = sub.groupby(['feature1','feature3'])['speed_wpm'].mean().unstack()
print("participants total", agg.shape)

# paired participants with both conditions
paired = agg.dropna()
print("paired participants", paired.shape)

if paired.shape[0] > 1:
    t = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
    print("paired t-test", t)
    # compute effect size (Cohen's d for paired)
    diff = paired[1] - paired[0]
    d = diff.mean() / diff.std(ddof=1)
    print("paired effect size d", d)

# also use all observations with mixed approach via simple t-test (not ideal)
# independent t-test
s1 = sub[sub['feature3']==1]['speed_wpm']
s0 = sub[sub['feature3']==0]['speed_wpm']
# drop inf
s1 = s1.replace([np.inf,-np.inf], np.nan).dropna()
s0 = s0.replace([np.inf,-np.inf], np.nan).dropna()
print("obs counts", len(s1), len(s0))
if len(s1)>1 and len(s0)>1:
    t2 = stats.ttest_ind(s1, s0, equal_var=False)
    print("independent t-test", t2)
    # effect size (Cohen's d)
    # pooled sd with unequal sizes
    d2 = (s1.mean() - s0.mean()) / np.sqrt(((s1.std(ddof=1)**2)+(s0.std(ddof=1)**2))/2)
    print("independent d", d2)

# check if feature20 works similarly for speed
sub2 = _df[_df['dyslexia_any']].copy()
agg2 = sub2.groupby(['feature1','feature3'])['feature20'].mean().unstack()
paired2 = agg2.dropna()
print("paired feature20", paired2.shape)
if paired2.shape[0] > 1:
    t3 = stats.ttest_rel(paired2[1], paired2[0], nan_policy='omit')
    diff2 = paired2[1]-paired2[0]
    d3 = diff2.mean()/diff2.std(ddof=1)
    print("paired t-test feature20", t3)
    print("paired effect size d (feature20)", d3)

