import pandas as pd

_df = pd.read_csv('amtl.csv')
print(_df[['genus','age','pop','num_amtl','stdev_age']].corr())
