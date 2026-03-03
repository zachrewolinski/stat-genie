import pandas as pd

df = pd.read_csv('amtl.csv')
# pick first specimen id
spec = df['prob_male'].iloc[0]
sub = df[df['prob_male'] == spec]
print('specimen', spec)
print(sub)
print('\nunique per column in specimen:')
for col in df.columns:
    print(col, sub[col].nunique())
