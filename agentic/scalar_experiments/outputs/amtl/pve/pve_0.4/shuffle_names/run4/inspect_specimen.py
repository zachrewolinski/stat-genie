import pandas as pd

df = pd.read_csv('amtl.csv')

# pick a specimen
spec = df['prob_male'].unique()[0]
print('specimen', spec)
print(df[df['prob_male']==spec])

# another specimen with negative genus
spec_neg = df.loc[df['genus']<0, 'prob_male'].iloc[0]
print('\nnegative specimen', spec_neg)
print(df[df['prob_male']==spec_neg])
