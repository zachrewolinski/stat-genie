import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Basic checks
print('rows', len(df))
print('columns', df.columns.tolist())

# Define dyslexia group
# Use dyslexia_bin if available; also check dyslexia values
if 'dyslexia_bin' in df.columns:
    df['dyslexia_group'] = df['dyslexia_bin']
else:
    df['dyslexia_group'] = (df['dyslexia'] > 0).astype(int)

# Subset to dyslexic participants
sub = df[df['dyslexia_group'] == 1].copy()
print('dyslexic rows', len(sub))

# Check reader_view values
print('reader_view counts', sub['reader_view'].value_counts(dropna=False).to_dict())

# Basic descriptive stats by reader_view
summary = sub.groupby('reader_view')['speed'].agg(['count','mean','median','std'])
print('\nSummary speed by reader_view (dyslexic):')
print(summary)

# Compute t-test (Welch) and Mann-Whitney
rv1 = sub[sub['reader_view'] == 1]['speed'].dropna()
rv0 = sub[sub['reader_view'] == 0]['speed'].dropna()

# Welch t-test
if len(rv1) > 1 and len(rv0) > 1:
    tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False)
    print('\nWelch t-test: t=', tstat, 'p=', pval)

    # Mann-Whitney U test
    try:
        ustat, pval_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
        print('Mann-Whitney U: U=', ustat, 'p=', pval_u)
    except Exception as e:
        print('Mann-Whitney error', e)

    # Cohen's d
    # Pooled standard deviation (unequal sizes)
    n1, n0 = len(rv1), len(rv0)
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    pooled = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2)/(n1+n0-2))
    d = (rv1.mean() - rv0.mean())/pooled
    print('Cohen d:', d)

# Check if within-subject comparison possible
# For each uuid, check if both reader_view conditions appear
if 'uuid' in sub.columns:
    per_uuid = sub.groupby('uuid')['reader_view'].nunique()
    both = per_uuid[per_uuid >= 2]
    print('\nParticipants with both conditions:', len(both))
    if len(both) > 5:
        # Paired comparison by participant: mean speed per condition per uuid
        pivot = sub.groupby(['uuid','reader_view'])['speed'].mean().unstack()
        pivot = pivot.dropna()
        if 0 in pivot.columns and 1 in pivot.columns and len(pivot) > 1:
            tstat_p, pval_p = stats.ttest_rel(pivot[1], pivot[0])
            print('Paired t-test by uuid mean speed: t=', tstat_p, 'p=', pval_p)
            # Wilcoxon signed-rank
            try:
                wstat, pval_w = stats.wilcoxon(pivot[1], pivot[0])
                print('Wilcoxon signed-rank: W=', wstat, 'p=', pval_w)
            except Exception as e:
                print('Wilcoxon error', e)
            # effect size for paired: Cohen d for paired (mean diff / sd diff)
            diff = pivot[1] - pivot[0]
            d_p = diff.mean()/diff.std(ddof=1)
            print('Paired Cohen d:', d_p)

# Explore log speed to address skew
sub = sub.copy()
sub['log_speed'] = np.log(sub['speed'])
rv1_log = sub[sub['reader_view'] == 1]['log_speed'].dropna()
rv0_log = sub[sub['reader_view'] == 0]['log_speed'].dropna()
if len(rv1_log) > 1 and len(rv0_log) > 1:
    tstat_log, pval_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False)
    print('\nWelch t-test on log(speed): t=', tstat_log, 'p=', pval_log)

# Summary for log speed
summary_log = sub.groupby('reader_view')['log_speed'].agg(['count','mean','median','std'])
print('\nSummary log(speed) by reader_view (dyslexic):')
print(summary_log)

