import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

wc = df['feature7']
ft = df['feature4']
fr = df['feature5']
fs = df['feature6']
rs = df['feature20']

# Candidate computations
speed_total = wc * 60000.0 / ft
speed_reading = wc * 60000.0 / fr
speed_scroll = wc * 60000.0 / fs.replace(0, np.nan)

# Time per word (ms per word)
ms_per_word_total = ft / wc
ms_per_word_read = fr / wc

# Correlations
candidates = {
    'wpm_total': speed_total,
    'wpm_read': speed_reading,
    'wpm_scroll': speed_scroll,
    'ms_per_word_total': ms_per_word_total,
    'ms_per_word_read': ms_per_word_read,
}

for name, series in candidates.items():
    valid = np.isfinite(rs) & np.isfinite(series)
    corr = np.corrcoef(rs[valid], series[valid])[0,1]
    print(name, corr)
