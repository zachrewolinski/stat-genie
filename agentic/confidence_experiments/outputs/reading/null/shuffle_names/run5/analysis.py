import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# Map columns based on info.json descriptions
participant_id = 'speed'  # unique id
reader_view_col = 'language'  # 0/1 indicator
page_id_col = 'scrolling_time'  # page id
# time columns (ms)
page_time_col = 'adjusted_running_time'  # time on page
reading_time_col = 'age'  # time on page minus scrolling
scrolling_time_col = 'gender'  # scrolling time
# word count and img width
num_words_col = 'retake_trial'  # number of words
img_width_col = 'num_words'  # image width
# dyslexia status
# device col is dyslexia status (0 no, 1 dyslexia, 2 severe)
dyslexia_status_col = 'device'
# binary dyslexia indicator column
# correct_rate col indicates dyslexia (1/0)
dyslexia_bin_col = 'correct_rate'
# reading speed candidate
speed_col_candidate = 'running_time'

# Basic checks
summary = {}
summary['n_rows'] = len(df)
summary['n_participants'] = df[participant_id].nunique()
summary['reader_view_counts'] = df[reader_view_col].value_counts(dropna=False).to_dict()
summary['dyslexia_status_counts'] = df[dyslexia_status_col].value_counts(dropna=False).to_dict()
summary['dyslexia_bin_counts'] = df[dyslexia_bin_col].value_counts(dropna=False).to_dict()

# Compute reading speed from words / reading_time (ms)
# avoid division by zero
speed_from_reading = df[num_words_col] / (df[reading_time_col] / 60000.0)

# Compare with running_time column
valid = speed_from_reading.replace([np.inf, -np.inf], np.nan).dropna()
rt_valid = df.loc[valid.index, speed_col_candidate]
if len(valid) > 0:
    corr = valid.corr(rt_valid)
else:
    corr = np.nan
summary['corr_computed_vs_running_time'] = corr
summary['computed_speed_stats'] = valid.describe().to_dict()
summary['running_time_stats'] = df[speed_col_candidate].describe().to_dict()

# Choose reading speed measure
# If correlation is high (>0.9), use running_time as reading speed; otherwise use computed speed
use_running_time = (corr is not np.nan) and (corr > 0.9)
summary['use_running_time_as_speed'] = use_running_time

if use_running_time:
    df['reading_speed'] = df[speed_col_candidate]
else:
    df['reading_speed'] = speed_from_reading

# Define dyslexic group (device >=1)
dyslexic_mask = df[dyslexia_status_col] >= 1

# Also define alt dyslexic mask from binary column
alt_dyslexic_mask = df[dyslexia_bin_col] == 1

# Function to compute stats and t-test

def compare_groups(data, mask, group_col=reader_view_col, value_col='reading_speed'):
    sub = data[mask].copy()
    sub = sub[[group_col, value_col, participant_id]].dropna()

    # per-record comparison
    g0 = sub[sub[group_col] == 0][value_col]
    g1 = sub[sub[group_col] == 1][value_col]

    # Welch t-test
    if len(g0) > 1 and len(g1) > 1:
        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    else:
        t_stat, p_val = np.nan, np.nan

    # effect size (Cohen's d, using pooled SD)
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

    d = cohens_d(g1, g0)

    # per-participant means to reduce repeated measures
    participant_means = (
        sub.groupby([participant_id, group_col])[value_col]
        .mean()
        .reset_index()
    )
    g0_p = participant_means[participant_means[group_col] == 0][value_col]
    g1_p = participant_means[participant_means[group_col] == 1][value_col]
    if len(g0_p) > 1 and len(g1_p) > 1:
        t_stat_p, p_val_p = stats.ttest_ind(g1_p, g0_p, equal_var=False, nan_policy='omit')
    else:
        t_stat_p, p_val_p = np.nan, np.nan
    d_p = cohens_d(g1_p, g0_p)

    result = {
        'n_total': len(sub),
        'n_reader_view_0': len(g0),
        'n_reader_view_1': len(g1),
        'mean_rv0': float(np.nanmean(g0)) if len(g0) else np.nan,
        'mean_rv1': float(np.nanmean(g1)) if len(g1) else np.nan,
        't_stat': float(t_stat) if np.isfinite(t_stat) else np.nan,
        'p_val': float(p_val) if np.isfinite(p_val) else np.nan,
        'cohens_d': float(d) if np.isfinite(d) else np.nan,
        'participant_n_rv0': len(g0_p),
        'participant_n_rv1': len(g1_p),
        'mean_rv0_participant': float(np.nanmean(g0_p)) if len(g0_p) else np.nan,
        'mean_rv1_participant': float(np.nanmean(g1_p)) if len(g1_p) else np.nan,
        't_stat_participant': float(t_stat_p) if np.isfinite(t_stat_p) else np.nan,
        'p_val_participant': float(p_val_p) if np.isfinite(p_val_p) else np.nan,
        'cohens_d_participant': float(d_p) if np.isfinite(d_p) else np.nan,
    }
    return result


results_device = compare_groups(df, dyslexic_mask)
results_binary = compare_groups(df, alt_dyslexic_mask)

print('SUMMARY')
print(summary)
print('RESULTS_DEVICE')
print(results_device)
print('RESULTS_BINARY')
print(results_binary)
