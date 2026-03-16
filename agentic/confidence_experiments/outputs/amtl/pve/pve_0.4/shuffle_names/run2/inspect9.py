import pandas as pd

_df = pd.read_csv('amtl.csv')

prop = _df['genus'] / _df['age']
print('prop min', prop.min(), 'max', prop.max())
print('frac prop <0', (prop<0).mean(), 'frac >1', (prop>1).mean())

# clip genus to [0, age] and see differences
clip = _df['genus'].clip(lower=0)
prop_clip = clip / _df['age']
print('after clip, prop min', prop_clip.min(), 'max', prop_clip.max(), 'frac >1', (prop_clip>1).mean())

