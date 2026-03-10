import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('reading.csv')

# Filter to dyslexic individuals (dyslexia_bin == 1)
dyslexic = df[df['dyslexia_bin'] == 1].copy()

# Basic counts
n_total = len(df)
n_dys = len(dyslexic)
n_participants = dyslexic['uuid'].nunique()

# Group summary by reader_view
group_stats = dyslexic.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Check per-participant paired data (have both reader_view conditions)
per_uuid = dyslexic.groupby('uuid')['reader_view'].nunique()
n_paired = (per_uuid == 2).sum()
paired_uuids = per_uuid[per_uuid == 2].index

# For paired participants, compute within-subject mean speed for each condition
paired = dyslexic[dyslexic['uuid'].isin(paired_uuids)].copy()
paired_means = paired.groupby(['uuid', 'reader_view'])['speed'].mean().unstack()
paired_means = paired_means.dropna()

# Paired t-test on within-subject means
if len(paired_means) > 1:
    t_paired = stats.ttest_rel(paired_means[1], paired_means[0])
    diff = paired_means[1] - paired_means[0]
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
else:
    t_paired = None
    dz = np.nan

# Unpaired t-test (all observations) and Mann-Whitney
speed_rv1 = dyslexic[dyslexic['reader_view'] == 1]['speed']
speed_rv0 = dyslexic[dyslexic['reader_view'] == 0]['speed']

if len(speed_rv1) > 1 and len(speed_rv0) > 1:
    t_unpaired = stats.ttest_ind(speed_rv1, speed_rv0, equal_var=False, nan_policy='omit')
    mw = stats.mannwhitneyu(speed_rv1, speed_rv0, alternative='two-sided')
else:
    t_unpaired = None
    mw = None

# Log-transform speed for skewness (add small constant to avoid log(0))
dyslexic['log_speed'] = np.log(dyslexic['speed'] + 1e-6)

# Mixed-effects model: log_speed ~ reader_view + (1|uuid)
try:
    model = smf.mixedlm("log_speed ~ reader_view", dyslexic, groups=dyslexic["uuid"])
    mres = model.fit(reml=False)
    mixed_p = mres.pvalues.get('reader_view', np.nan)
    mixed_coef = mres.params.get('reader_view', np.nan)
except Exception:
    mres = None
    mixed_p = np.nan
    mixed_coef = np.nan

# Print results
print('Total rows:', n_total)
print('Dyslexic rows:', n_dys)
print('Dyslexic participants:', n_participants)
print('Group stats by reader_view (dyslexic):')
print(group_stats)

print('\nPaired participants with both conditions:', n_paired)
if t_paired is not None:
    print('Paired t-test on within-subject means: t=%.4f p=%.6f' % (t_paired.statistic, t_paired.pvalue))
    print('Paired Cohen dz:', dz)

if t_unpaired is not None:
    print('\nUnpaired t-test (Welch): t=%.4f p=%.6f' % (t_unpaired.statistic, t_unpaired.pvalue))
    print('Mann-Whitney U: U=%.1f p=%.6f' % (mw.statistic, mw.pvalue))

if mres is not None:
    print('\nMixedLM log_speed ~ reader_view + (1|uuid):')
    print('coef (reader_view):', mixed_coef)
    print('p-value (reader_view):', mixed_p)
    # Print table of coefficients for quick view
    print(mres.summary().tables[1])
else:
    print('\nMixedLM failed')

# Effect size Cohen's d for unpaired

def cohens_d(x, y):
    x = np.array(x)
    y = np.array(y)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp if sp != 0 else np.nan

if len(speed_rv1) > 1 and len(speed_rv0) > 1:
    d_unpaired = cohens_d(speed_rv1, speed_rv0)
    print('\nUnpaired Cohen d (speed):', d_unpaired)

mean_rv1 = speed_rv1.mean() if len(speed_rv1) > 0 else np.nan
mean_rv0 = speed_rv0.mean() if len(speed_rv0) > 0 else np.nan
print('Mean speed rv=1:', mean_rv1)
print('Mean speed rv=0:', mean_rv0)
print('Percent difference (rv1 vs rv0):', (mean_rv1 - mean_rv0) / mean_rv0 * 100 if mean_rv0 not in [0, np.nan] else np.nan)
