import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Identify dyslexic participants
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1]
else:
    dys = df[df['dyslexia'].isin([1,2])]

# Ensure speed positive

dys = dys.copy()

dys = dys[dys['speed'].notna()]

# remove non-positive speeds for log

pos = dys[dys['speed'] > 0].copy()

# group by reader_view

g0 = pos[pos['reader_view'] == 0]

g1 = pos[pos['reader_view'] == 1]


def summarize(group):
    return {
        'n': len(group),
        'mean': group['speed'].mean(),
        'median': group['speed'].median(),
        'std': group['speed'].std(),
    }

summary = {'no_reader_view': summarize(g0), 'reader_view': summarize(g1)}

# Welch t-test on log speed

log0 = np.log(g0['speed'])
log1 = np.log(g1['speed'])


t_stat, p_val = stats.ttest_ind(log1, log0, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)

u_stat, p_u = stats.mannwhitneyu(g1['speed'], g0['speed'], alternative='two-sided')

# Cohen's d on log speed

mean0 = log0.mean()
mean1 = log1.mean()
std0 = log0.std(ddof=1)
std1 = log1.std(ddof=1)

pooled_sd = np.sqrt(((len(log0)-1)*std0**2 + (len(log1)-1)*std1**2) / (len(log0)+len(log1)-2))

cohen_d = (mean1 - mean0) / pooled_sd

# percent difference on geometric mean

geom0 = np.exp(mean0)
geom1 = np.exp(mean1)

pct_diff = (geom1 - geom0) / geom0 * 100


print('Dyslexic subset size:', len(dys))
print('Positive speed size:', len(pos))
print('Summary:', summary)
print('Welch t-test on log speed: t=%.3f p=%.6f' % (t_stat, p_val))
print('Mann-Whitney U on speed: U=%.3f p=%.6f' % (u_stat, p_u))
print('Cohen d (log speed): %.3f' % cohen_d)
print('Geometric mean speed no reader view:', geom0)
print('Geometric mean speed reader view:', geom1)
print('Percent diff (geom): %.2f%%' % pct_diff)

