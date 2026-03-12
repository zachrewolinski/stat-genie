import pandas as pd

_df = pd.read_csv('amtl.csv')
print(_df[['genus','num_amtl','age','pop','stdev_age']].corr())
