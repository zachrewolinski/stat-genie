import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# sum genus across sockets within specimen
sum_genus = df.groupby('prob_male')['genus'].sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()

# compare
diff = sum_genus - num_amtl
print('Mean diff', diff.mean(), 'std', diff.std(), 'min', diff.min(), 'max', diff.max())
print('Correlation between sum_genus and num_amtl', sum_genus.corr(num_amtl))

# show sample
print(pd.DataFrame({'sum_genus':sum_genus.head(), 'num_amtl':num_amtl.head(), 'diff':diff.head()}))

