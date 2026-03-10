import json
import numpy as np
import pandas as pd
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Map variables based on metadata (names are shuffled)
# language: reader view indicator (1 on, 0 off)
# running_time: reading speed (wpm-like; highly correlated with words/min computed from num_words & adjusted_running_time)
# correct_rate: dyslexia indicator (1 dyslexia, 0 no dyslexia)

reader_view = _df['language']
reading_speed = _df['running_time']

# Dyslexia group
mask_dys = _df['correct_rate'] == 1

# Trial-level comparison
rv1 = reading_speed[mask_dys & (reader_view == 1)]
rv0 = reading_speed[mask_dys & (reader_view == 0)]

# Welch t-test
welch_t, welch_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Effect size (Cohen's d)
n1, n0 = rv1.size, rv0.size
s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
cohen_d = (rv1.mean() - rv0.mean()) / sp if sp > 0 else np.nan

# Participant-level paired comparison
sub = _df.loc[mask_dys, ['speed', 'language', 'running_time']].dropna()
paired = sub.pivot_table(index='speed', columns='language', values='running_time', aggfunc='mean').dropna()

paired_t, paired_p = stats.ttest_1samp(paired[1] - paired[0], 0)

# Paired effect size (Cohen's dz)
diff = paired[1] - paired[0]
cohen_dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

# Summary stats
mean_rv1 = rv1.mean()
mean_rv0 = rv0.mean()
mean_diff = mean_rv1 - mean_rv0

n_trials = int(mask_dys.sum())
n_participants = int(paired.shape[0])

# Decide Likert response: no evidence of improvement, small negative effect
response = 25

explanation = (
    "Using the dyslexia indicator (correct_rate=1), there are {n_trials} dyslexic trials from {n_participants} participants, "
    "with balanced Reader View conditions (n={n1} on, n={n0} off). "
    "Mean reading speed (running_time) is {mean_rv1:.1f} with Reader View on vs {mean_rv0:.1f} off, "
    "a difference of {mean_diff:.1f} (on-off). "
    "The Welch t-test shows no significant difference (t={welch_t:.3f}, p={welch_p:.3f}) and the effect size is tiny "
    "(Cohen's d={cohen_d:.3f}). A participant-level paired test also shows no significant improvement "
    "(t={paired_t:.3f}, p={paired_p:.3f}, Cohen's dz={cohen_dz:.3f}). "
    "Overall, there is no evidence that Reader View improves reading speed for individuals with dyslexia in this dataset; "
    "if anything, speeds are slightly lower with Reader View." 
).format(
    n_trials=n_trials,
    n_participants=n_participants,
    n1=n1,
    n0=n0,
    mean_rv1=mean_rv1,
    mean_rv0=mean_rv0,
    mean_diff=mean_diff,
    welch_t=welch_t,
    welch_p=welch_p,
    cohen_d=cohen_d,
    paired_t=paired_t,
    paired_p=paired_p,
    cohen_dz=cohen_dz,
)

with open('conclusion.txt', 'w') as f:
    json.dump({"response": response, "explanation": explanation}, f)

