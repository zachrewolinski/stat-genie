import pandas as pd
import numpy as np

DF = pd.read_csv('reading.csv')
print('rows', DF.shape)
print('columns', DF.columns.tolist())
print('dyslexia_bin counts', DF['dyslexia_bin'].value_counts(dropna=False))
print('reader_view counts', DF['reader_view'].value_counts(dropna=False))
print('speed summary', DF['speed'].describe())

# dyslexia subset
sub = DF[DF['dyslexia_bin'] == 1]
print('dyslexia rows', sub.shape)
print('dyslexia reader_view counts', sub['reader_view'].value_counts())
print('dyslexia speed summary', sub['speed'].describe())

# per uuid counts
print('unique uuids', DF['uuid'].nunique())
print('unique uuids dyslexia', sub['uuid'].nunique())

