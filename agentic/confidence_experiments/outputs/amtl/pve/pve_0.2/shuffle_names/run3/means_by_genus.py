import pandas as pd

_df = pd.read_csv('amtl.csv')
# genus category is in tooth_class column
print(_df.groupby('tooth_class')['genus'].agg(['mean','median','std','count']).sort_values('mean', ascending=False))
