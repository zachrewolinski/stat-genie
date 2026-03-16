import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('reading.csv')

participant_id = 'speed'
reader_view_col = 'language'
reading_time_col = 'age'  # ms
word_count_col = 'retake_trial'
dyslexia_status_col = 'device'

# compute reading speed (words per minute)
df['reading_speed'] = df[word_count_col] / (df[reading_time_col] / 60000.0)

# dyslexic = device >= 1
sub = df[df[dyslexia_status_col] >= 1].copy()
sub = sub[[participant_id, reader_view_col, 'reading_speed']].dropna()

# per-record stats
rv0 = sub[sub[reader_view_col] == 0]['reading_speed']
rv1 = sub[sub[reader_view_col] == 1]['reading_speed']

# welch t-test
if len(rv0) > 1 and len(rv1) > 1:
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    t_stat, p_val = np.nan, np.nan

# mann-whitney
if len(rv0) > 0 and len(rv1) > 0:
    try:
        u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    except Exception:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

# participant-level means
participant_means = sub.groupby([participant_id, reader_view_col])['reading_speed'].mean().reset_index()
rv0_p = participant_means[participant_means[reader_view_col] == 0]['reading_speed']
rv1_p = participant_means[participant_means[reader_view_col] == 1]['reading_speed']
if len(rv0_p) > 1 and len(rv1_p) > 1:
    t_stat_p, p_val_p = stats.ttest_ind(rv1_p, rv0_p, equal_var=False, nan_policy='omit')
else:
    t_stat_p, p_val_p = np.nan, np.nan

# effect size

def cohens_d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


d = cohens_d(rv1, rv0)
d_p = cohens_d(rv1_p, rv0_p)

print('n_dyslexic_records', len(sub))
print('n_rv0', len(rv0), 'n_rv1', len(rv1))
print('mean_rv0', rv0.mean(), 'mean_rv1', rv1.mean())
print('median_rv0', rv0.median(), 'median_rv1', rv1.median())
print('t_stat', t_stat, 'p_val', p_val, 'cohens_d', d)
print('mannwhitney_p', p_u)
print('participant_n_rv0', len(rv0_p), 'participant_n_rv1', len(rv1_p))
print('mean_rv0_p', rv0_p.mean(), 'mean_rv1_p', rv1_p.mean())
print('t_stat_p', t_stat_p, 'p_val_p', p_val_p, 'cohens_d_p', d_p)
