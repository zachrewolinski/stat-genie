import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('reading.csv')

# candidate derived metrics
speed4 = 60000 * df['feature7'] / df['feature4']
speed5 = 60000 * df['feature7'] / df['feature5']
ms_per_word4 = df['feature4'] / df['feature7']
ms_per_word5 = df['feature5'] / df['feature7']

candidates = {
    'speed4': speed4,
    'speed5': speed5,
    'ms_per_word4': ms_per_word4,
    'ms_per_word5': ms_per_word5,
}

for name, series in candidates.items():
    corr = np.corrcoef(series, df['feature20'])[0,1]
    print(name, corr)

# maybe feature20 is log speed? check correlation with log
for name, series in candidates.items():
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series[series>0]
    feat20 = df.loc[series.index, 'feature20']
    # log both
    corr = np.corrcoef(np.log(series), np.log(feat20))[0,1]
    print('log', name, corr)
