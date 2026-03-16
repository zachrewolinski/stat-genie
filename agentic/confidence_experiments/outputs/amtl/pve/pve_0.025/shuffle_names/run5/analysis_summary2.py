import pandas as pd
_df = pd.read_csv('amtl.csv')
_df['is_human'] = (_df['tooth_class'] == 'Homo sapiens').astype(int)
print('human mean', _df[_df['is_human']==1]['genus'].mean())
print('nonhuman mean', _df[_df['is_human']==0]['genus'].mean())
