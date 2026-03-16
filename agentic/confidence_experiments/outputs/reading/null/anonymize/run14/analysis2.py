import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# candidate speed metrics
ms_per_word = df['feature5'] / df['feature7']
words_per_sec = df['feature7'] / (df['feature5'] / 1000.0)
words_per_min = df['feature7'] / (df['feature5'] / 60000.0)

print('corr feature20 vs ms_per_word', np.corrcoef(df['feature20'], ms_per_word)[0,1])
print('corr feature20 vs words_per_sec', np.corrcoef(df['feature20'], words_per_sec)[0,1])
print('corr feature20 vs words_per_min', np.corrcoef(df['feature20'], words_per_min)[0,1])

# check scale: typical words per minute should be around 200-300. feature20 median 287.5, so maybe wpm. But then should correlate with words per minute derived from feature5 and words; compute difference.
print('wpm derived describe', words_per_min.describe())
print('feature20 describe', df['feature20'].describe())
print('median ratio feature20/derived', (df['feature20']/words_per_min).median())
print('corr log', np.corrcoef(np.log(df['feature20']+1), np.log(words_per_min+1))[0,1])

# check if feature20 equals some constant / feature5 or feature4
for base in ['feature4','feature5']:
    inv = 60000.0 / df[base]
    print('corr feature20 vs 60000/'+base, np.corrcoef(df['feature20'], inv)[0,1])

# correlation with scroll time and comprehension maybe
for col in ['feature6','feature8','feature19']:
    print('corr feature20 vs', col, np.corrcoef(df['feature20'], df[col])[0,1])
