import pandas as pd

_df = pd.read_csv('amtl.csv')
# count tooth_class categories per specimen id
counts = _df.groupby('prob_male')['sockets'].nunique()
print(counts.value_counts().sort_index())
print('min', counts.min(), 'max', counts.max())
