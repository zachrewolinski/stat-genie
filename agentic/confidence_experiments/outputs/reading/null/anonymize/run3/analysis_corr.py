import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

for col in ['feature4','feature5','feature6','feature7']:
    corr = np.corrcoef(df[col], df['feature20'])[0,1]
    print('corr feature20 with', col, corr)

# maybe feature20 is reading speed in words/min but log? check log
for col in ['feature4','feature5']:
    corr = np.corrcoef(np.log(df['feature20']), df[col])[0,1]
    print('corr log(feature20) with', col, corr)

# check if feature20 maybe reciprocal of time per word
for col in ['feature4','feature5']:
    time_per_word = df[col]/df['feature7']
    inv = 60000/time_per_word
    corr = np.corrcoef(inv, df['feature20'])[0,1]
    print('corr feature20 with inv time/word from', col, corr)

