import pandas as pd
import numpy as np


df = pd.read_csv('reading.csv')

# computed speeds
speed_excl = df['feature7'] / (df['feature5'] / 1000.0) * 60.0
speed_incl = df['feature7'] / (df['feature4'] / 1000.0) * 60.0

# Replace inf
speed_excl = speed_excl.replace([np.inf, -np.inf], np.nan)
speed_incl = speed_incl.replace([np.inf, -np.inf], np.nan)


def describe(s):
    s = s.dropna()
    return {
        'n': int(s.shape[0]),
        'min': float(s.min()),
        'p1': float(s.quantile(0.01)),
        'p5': float(s.quantile(0.05)),
        'p25': float(s.quantile(0.25)),
        'median': float(s.quantile(0.5)),
        'p75': float(s.quantile(0.75)),
        'p95': float(s.quantile(0.95)),
        'p99': float(s.quantile(0.99)),
        'max': float(s.max())
    }

print('speed_excl', describe(speed_excl))
print('speed_incl', describe(speed_incl))

if 'feature20' in df.columns:
    f20 = df['feature20']
    print('feature20', describe(f20))

# check if feature20 maybe time per word or per page
# compute time per word in ms

time_per_word_ms = df['feature5'] / df['feature7']
print('time_per_word_ms', describe(time_per_word_ms))

# correlation between feature20 and time_per_word_ms
import numpy as np

corr_time_word = df[['feature20']].join(time_per_word_ms.rename('tpw')).corr().iloc[0,1]
print('corr feature20 vs time_per_word_ms', corr_time_word)

# correlation between feature20 and feature5/feature7 (seconds per word)
sec_per_word = time_per_word_ms/1000
corr_spw = df[['feature20']].join(sec_per_word.rename('spw')).corr().iloc[0,1]
print('corr feature20 vs sec_per_word', corr_spw)

# correlation between feature20 and words per minute
corr_wpm = df[['feature20']].join(speed_excl.rename('wpm')).corr().iloc[0,1]
print('corr feature20 vs wpm', corr_wpm)

# check if feature20 could be time per word * 1000 or something; compute best correlation with transformations

# log transform
log_f20 = np.log(df['feature20'].replace(0, np.nan))
log_wpm = np.log(speed_excl.replace(0, np.nan))
print('corr log feature20 vs log wpm', pd.concat([log_f20, log_wpm], axis=1).corr().iloc[0,1])

