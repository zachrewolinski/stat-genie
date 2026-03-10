import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

speed_wpm_f5 = df['feature7']*60000/df['feature5']
speed_wpm_f4 = df['feature7']*60000/df['feature4']
ms_per_word_f5 = df['feature5']/df['feature7']
ms_per_word_f4 = df['feature4']/df['feature7']

for name, series in [('speed_wpm_f5', speed_wpm_f5),('speed_wpm_f4', speed_wpm_f4),('ms_per_word_f5', ms_per_word_f5),('ms_per_word_f4', ms_per_word_f4)]:
    corr = np.corrcoef(series, df['feature20'])[0,1]
    print(name, 'corr', corr, 'desc', series.describe())

# check if feature20 maybe log or inverse of speed
inv_feature20 = 1/df['feature20']
for name, series in [('inv_feature20', inv_feature20)]:
    print('inv_feature20 desc', series.describe())

# maybe feature20 already speed in wpm computed from feature4? check mean
print('feature20 desc', df['feature20'].describe())

# check correlation of feature20 with reading time per word
corr_time = np.corrcoef(df['feature5'], df['feature20'])[0,1]
print('corr feature20 with feature5', corr_time)

