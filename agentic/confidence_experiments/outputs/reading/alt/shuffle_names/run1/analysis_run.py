import pandas as pd
import numpy as np
from scipy import stats

path = 'reading.csv'
df = pd.read_csv(path)

# Map columns based on value patterns
participant_id = 'speed'          # uuid-like
reader_view = 'language'          # binary 0/1, balanced
reading_speed = 'running_time'    # matches wpm derived from word_count / adjusted_time
# dyslexia status is likely the 0/1/2 numeric column NOT the gender-like distribution
# 'device' has 0/1/2 with mostly 0; 'dyslexia' has 0/1/2 with mostly 1 (likely gender)
# treat device as dyslexia status

dyslexia_status = 'device'

# Filter dyslexic participants (1 or 2)
sub = df[df[dyslexia_status].isin([1.0, 2.0])].copy()

# Ensure reader_view is 0/1
sub = sub[sub[reader_view].isin([0, 1])]

# Compute per-participant mean speed by condition
pivot = (
    sub.pivot_table(index=participant_id, columns=reader_view, values=reading_speed, aggfunc='mean')
)

# Keep participants with both conditions
paired = pivot.dropna()

n_participants = paired.shape[0]

# Paired t-test
if n_participants > 1:
    diff = paired[1] - paired[0]
    t_stat, p_value = stats.ttest_rel(paired[1], paired[0])
    # effect size (paired Cohen's d)
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
    mean0 = paired[0].mean()
    mean1 = paired[1].mean()
    mean_diff = diff.mean()
    # Wilcoxon signed-rank as robustness
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(paired[1], paired[0])
    except ValueError:
        p_wilcoxon = np.nan
else:
    t_stat = p_value = d = mean0 = mean1 = mean_diff = np.nan
    p_wilcoxon = np.nan

# Also compute unpaired comparison at row level (Welch) as secondary check
rv0 = sub[sub[reader_view] == 0][reading_speed]
rv1 = sub[sub[reader_view] == 1][reading_speed]

if len(rv0) > 1 and len(rv1) > 1:
    t_stat_u, p_value_u = stats.ttest_ind(rv1, rv0, equal_var=False)
    mean0_u = rv0.mean(); mean1_u = rv1.mean(); mean_diff_u = mean1_u - mean0_u
else:
    t_stat_u = p_value_u = mean0_u = mean1_u = mean_diff_u = np.nan

print('Dyslexic rows:', len(sub))
print('Participants with both conditions:', n_participants)
print('Paired means: rv0', mean0, 'rv1', mean1, 'diff', mean_diff)
print('Paired t-test p=', p_value, 't=', t_stat, 'd=', d)
print('Wilcoxon p=', p_wilcoxon)
print('Unpaired means: rv0', mean0_u, 'rv1', mean1_u, 'diff', mean_diff_u)
print('Unpaired Welch p=', p_value_u)
