import pandas as pd
import numpy as np
from scipy import stats

path='reading.csv'
df=pd.read_csv(path)

# Identify columns based on inspection
reader_view_col = 'language'  # binary 0/1, varies within participants
speed_col = 'speed'           # participant id
reading_speed_col = 'running_time'  # plausible reading speed measure
# dyslexia status from device column (0 no, 1 dyslexia, 2 severe)
dyslexia_col = 'device'

# Determine participant-level dyslexia status by mode

def mode_nonan(series):
    s = series.dropna()
    if s.empty:
        return np.nan
    # if multiple modes, take the max (more severe) to be conservative
    modes = s.mode()
    return modes.max()

participant_dys = df.groupby(speed_col)[dyslexia_col].apply(mode_nonan)

# Dyslexic participants: mode >=1
dys_participants = participant_dys[participant_dys >= 1].index

# Filter data for dyslexic participants
sub = df[df[speed_col].isin(dys_participants)].copy()

# Drop missing reader_view or reading_speed
sub = sub.dropna(subset=[reader_view_col, reading_speed_col])

# Ensure reader_view is 0/1
sub = sub[sub[reader_view_col].isin([0,1])]

# Participant-level means by reader_view
means = sub.groupby([speed_col, reader_view_col])[reading_speed_col].mean().unstack()

# Keep participants with both conditions
paired = means.dropna()

# Paired differences: reader_view=1 minus 0
if 0 in paired.columns and 1 in paired.columns:
    diff = paired[1] - paired[0]
else:
    diff = pd.Series(dtype=float)

results = {}
results['n_participants_dys'] = len(dys_participants)
results['n_paired'] = len(diff)

if len(diff) > 1:
    tstat, pval = stats.ttest_rel(paired[1], paired[0])
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    # 95% CI for mean diff
    ci_low, ci_high = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=sd_diff/np.sqrt(len(diff)))
    cohen_dz = mean_diff / sd_diff if sd_diff != 0 else np.nan
    results.update({
        'mean_diff': mean_diff,
        'sd_diff': sd_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'tstat': tstat,
        'pval': pval,
        'cohen_dz': cohen_dz,
        'mean_on': paired[1].mean(),
        'mean_off': paired[0].mean(),
    })

# Also compute independent-samples comparison across trials for dyslexic participants
on = sub[sub[reader_view_col]==1][reading_speed_col]
off = sub[sub[reader_view_col]==0][reading_speed_col]
results['n_trials_on'] = len(on)
results['n_trials_off'] = len(off)
if len(on) > 1 and len(off) > 1:
    tstat2, pval2 = stats.ttest_ind(on, off, equal_var=False)
    mean_on = on.mean(); mean_off = off.mean()
    # Hedges g
    n1, n2 = len(on), len(off)
    s1, s2 = on.var(ddof=1), off.var(ddof=1)
    s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2)) if n1+n2-2>0 else np.nan
    d = (mean_on - mean_off)/s_pooled if s_pooled and not np.isnan(s_pooled) else np.nan
    # small sample correction
    g = d*(1 - 3/(4*(n1+n2)-9)) if d==d else np.nan
    results.update({
        'mean_on_trial': mean_on,
        'mean_off_trial': mean_off,
        'tstat_ind': tstat2,
        'pval_ind': pval2,
        'hedges_g': g,
    })

print(results)
