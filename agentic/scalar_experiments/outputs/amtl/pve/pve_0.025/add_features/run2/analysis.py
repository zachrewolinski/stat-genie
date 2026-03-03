import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print('columns', df.columns.tolist())

# summary for num_amtl
num = df['num_amtl']
print('num_amtl mean', num.mean(), 'std', num.std(), 'min', num.min(), 'max', num.max())

# proportion check
within = ((num >= 0) & (num <= df['sockets'])).mean()
print('fraction num_amtl within [0, sockets]:', within)

# check integer-like
intlike = (np.isclose(num, np.round(num))).mean()
print('fraction num_amtl integer-like:', intlike)

# examine possible rate if num_amtl maybe standardized; compute rate = num_amtl / sockets
rate = num / df['sockets']
print('rate stats mean', rate.mean(), 'min', rate.min(), 'max', rate.max())

# group summary by genus: mean num_amtl, mean rate, count
summary = df.groupby('genus').agg(
    mean_num=('num_amtl','mean'),
    mean_rate=('num_amtl', lambda x: (x/df.loc[x.index, 'sockets']).mean()),
    n=('num_amtl','size')
)
print('summary by genus:\n', summary)

# check sockets distribution
print('sockets unique', sorted(df['sockets'].unique())[:10], '...')

# check tooth_class distribution by genus
print('tooth_class by genus counts')
print(pd.crosstab(df['genus'], df['tooth_class']))

