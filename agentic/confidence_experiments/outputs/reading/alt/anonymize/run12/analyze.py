import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.width', 200)

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Identify key variables
reader_view = 'feature3'  # 1=reader view
reading_speed = 'feature20'  # words per minute
participant_id = 'feature1'
dyslexia_flag = 'feature17'  # 1=dyslexia

# Filter dyslexic participants
sub = df[df[dyslexia_flag] == 1].copy()

# Basic counts
n_rows = len(sub)
participants = sub[participant_id].nunique()
print('dyslexia rows', n_rows)
print('dyslexia participants', participants)

# Check reader view counts
counts = sub[reader_view].value_counts().sort_index()
print('reader view counts (0,1):')
print(counts)

# Function for summary stats

def summary_stats(series):
    return {
        'n': int(series.count()),
        'mean': float(series.mean()),
        'std': float(series.std(ddof=1)),
        'median': float(series.median()),
    }

# Group comparison (raw rows)
rv0 = sub[sub[reader_view] == 0][reading_speed]
rv1 = sub[sub[reader_view] == 1][reading_speed]
print('raw summary rv0', summary_stats(rv0))
print('raw summary rv1', summary_stats(rv1))

# Welch t-test
welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('welch t-test', welch)

# Effect size (Cohen d for independent samples)
# use pooled SD (with unequal n)
mean_diff = rv1.mean() - rv0.mean()
var1 = rv1.var(ddof=1)
var0 = rv0.var(ddof=1)
n1 = rv1.count()
n0 = rv0.count()
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan
print('mean diff', mean_diff)
print('cohen d', cohen_d)

# Nonparametric test (Mann-Whitney U)
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('mannwhitney', mwu)
except ValueError as e:
    print('mannwhitney error', e)

# Participant-level paired analysis
# For each participant, compute mean speed per condition
pivot = sub.pivot_table(index=participant_id, columns=reader_view, values=reading_speed, aggfunc='mean')
paired = pivot.dropna()  # participants with both conditions
print('paired participants', paired.shape[0])

if paired.shape[0] > 1:
    diff = paired[1] - paired[0]
    print('paired diff summary', summary_stats(diff))
    ttest_paired = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
    print('paired t-test', ttest_paired)
    # effect size for paired: Cohen d_z (mean diff / std diff)
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    print('paired dz', dz)
else:
    print('not enough paired participants for paired test')

# Robust analysis: remove extreme outliers based on 99.5 percentile within dyslexia
cut = sub[reading_speed].quantile(0.995)
sub_trim = sub[sub[reading_speed] <= cut].copy()
rv0_trim = sub_trim[sub_trim[reader_view] == 0][reading_speed]
rv1_trim = sub_trim[sub_trim[reader_view] == 1][reading_speed]
print('trim cut', cut)
print('trim counts', rv0_trim.count(), rv1_trim.count())
print('trim summary rv0', summary_stats(rv0_trim))
print('trim summary rv1', summary_stats(rv1_trim))
welch_trim = stats.ttest_ind(rv1_trim, rv0_trim, equal_var=False, nan_policy='omit')
print('trim welch', welch_trim)

mean_diff_trim = rv1_trim.mean() - rv0_trim.mean()
var1t = rv1_trim.var(ddof=1)
var0t = rv0_trim.var(ddof=1)
nt1 = rv1_trim.count()
nt0 = rv0_trim.count()
pooled_sd_t = np.sqrt(((nt1 - 1) * var1t + (nt0 - 1) * var0t) / (nt1 + nt0 - 2))
cohen_d_t = mean_diff_trim / pooled_sd_t if pooled_sd_t > 0 else np.nan
print('trim mean diff', mean_diff_trim)
print('trim cohen d', cohen_d_t)

# Paired on trimmed
pivot_trim = sub_trim.pivot_table(index=participant_id, columns=reader_view, values=reading_speed, aggfunc='mean')
paired_trim = pivot_trim.dropna()
print('paired trimmed participants', paired_trim.shape[0])
if paired_trim.shape[0] > 1:
    diff_trim = paired_trim[1] - paired_trim[0]
    print('paired trim diff summary', summary_stats(diff_trim))
    ttest_paired_trim = stats.ttest_rel(paired_trim[1], paired_trim[0], nan_policy='omit')
    print('paired trim t-test', ttest_paired_trim)
    dz_trim = diff_trim.mean() / diff_trim.std(ddof=1) if diff_trim.std(ddof=1) > 0 else np.nan
    print('paired trim dz', dz_trim)
