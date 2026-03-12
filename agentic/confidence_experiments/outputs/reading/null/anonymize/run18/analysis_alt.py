import pandas as pd
import numpy as np
from scipy import stats


def analyze(df, dys_mask, time_col, label):
    df = df.copy()
    df['time_min'] = df[time_col] / 60000.0
    df['wpm'] = df['feature7'] / df['time_min']
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[(df['time_min'] > 0) & (df['feature7'] > 0)]
    df = df[df['feature3'].notna()]
    df = df[dys_mask]
    # trim 1/99 percentiles
    if len(df) > 0:
        lower = df['wpm'].quantile(0.01)
        upper = df['wpm'].quantile(0.99)
        df = df[(df['wpm'] >= lower) & (df['wpm'] <= upper)]
    on = df[df['feature3'] == 1]['wpm']
    off = df[df['feature3'] == 0]['wpm']
    # tests
    log_on = np.log(on)
    log_off = np.log(off)
    t_stat, p_val = stats.ttest_ind(log_on, log_off, equal_var=False, nan_policy='omit') if (len(log_on)>1 and len(log_off)>1) else (np.nan, np.nan)
    u_stat, p_u = stats.mannwhitneyu(on, off, alternative='two-sided') if (len(on)>0 and len(off)>0) else (np.nan, np.nan)
    return {
        'label': label,
        'n_on': int(len(on)),
        'n_off': int(len(off)),
        'mean_on': float(on.mean()),
        'mean_off': float(off.mean()),
        'median_on': float(on.median()),
        'median_off': float(off.median()),
        't_p': float(p_val),
        'u_p': float(p_u),
        'median_diff_pct': float((on.median() - off.median()) / off.median() * 100.0) if len(off)>0 else np.nan,
    }


df = pd.read_csv('reading.csv')

# Dyslexia masks
mask_feature17 = df['feature17'] == 1
mask_feature12 = df['feature12'].fillna(0) > 0

results = []
results.append(analyze(df, mask_feature17, 'feature5', 'dyslexia feature17, time feature5'))
results.append(analyze(df, mask_feature17, 'feature4', 'dyslexia feature17, time feature4'))
results.append(analyze(df, mask_feature12, 'feature5', 'dyslexia feature12>0, time feature5'))
results.append(analyze(df, mask_feature12, 'feature4', 'dyslexia feature12>0, time feature4'))

for r in results:
    print(r)

