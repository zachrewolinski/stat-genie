import json
import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Map variables based on info.json descriptions
# reader_view indicator (1/0)
reader_view = df['language']
# dyslexia status (0 no, 1 dyslexia, 2 severe)
dyslexia_status = df['device']
# num words on page
num_words = df['retake_trial']
# adjusted running time (ms) = time minus scrolling
time_adjusted_ms = df['age']
# total time on page (ms)
time_total_ms = df['adjusted_running_time']

# compute reading speed (words per minute)
# use adjusted time as primary (more direct reading time)
with np.errstate(divide='ignore', invalid='ignore'):
    speed_adj = num_words / (time_adjusted_ms / 60000.0)
    speed_total = num_words / (time_total_ms / 60000.0)

# build analysis frame
analysis_df = pd.DataFrame({
    'reader_view': reader_view,
    'dyslexia_status': dyslexia_status,
    'speed_adj': speed_adj,
    'speed_total': speed_total,
})

# filter to dyslexic participants (status > 0) and valid values
analysis_df = analysis_df.dropna(subset=['reader_view', 'dyslexia_status', 'speed_adj', 'speed_total'])
analysis_df = analysis_df[(analysis_df['dyslexia_status'] > 0) & np.isfinite(analysis_df['speed_adj']) & np.isfinite(analysis_df['speed_total'])]

# group by reader view
rv1 = analysis_df[analysis_df['reader_view'] == 1]
rv0 = analysis_df[analysis_df['reader_view'] == 0]

# helper: effect size

def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)

# primary metric: speed_adj
x = rv1['speed_adj']
y = rv0['speed_adj']

t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')

# nonparametric
u_stat, p_u = stats.mannwhitneyu(x, y, alternative='two-sided')

# effect size
cd = cohens_d(x, y)

# summary
summary = {
    'n_dyslexic': len(analysis_df),
    'n_reader_view_on': len(rv1),
    'n_reader_view_off': len(rv0),
    'mean_speed_adj_on': float(x.mean()),
    'mean_speed_adj_off': float(y.mean()),
    'median_speed_adj_on': float(x.median()),
    'median_speed_adj_off': float(y.median()),
    't_p_value': float(p_val),
    'mw_p_value': float(p_u),
    'cohens_d': float(cd),
}

# also check robustness with total time
x2 = rv1['speed_total']
y2 = rv0['speed_total']

t_stat2, p_val2 = stats.ttest_ind(x2, y2, equal_var=False, nan_policy='omit')
cd2 = cohens_d(x2, y2)
summary.update({
    'mean_speed_total_on': float(x2.mean()),
    'mean_speed_total_off': float(y2.mean()),
    't_p_value_total': float(p_val2),
    'cohens_d_total': float(cd2),
})

print(json.dumps(summary, indent=2))
