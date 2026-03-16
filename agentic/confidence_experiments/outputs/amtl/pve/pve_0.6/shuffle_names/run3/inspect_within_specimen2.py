import pandas as pd

df = pd.read_csv('amtl.csv')

spec_groups = df.groupby('prob_male')
print('stdev_age constant within specimen', (spec_groups['stdev_age'].nunique()==1).mean())
print('genus constant within specimen', (spec_groups['genus'].nunique()==1).mean())
print('age constant within specimen', (spec_groups['age'].nunique()==1).mean())

