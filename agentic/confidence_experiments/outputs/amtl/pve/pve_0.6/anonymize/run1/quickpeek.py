import pandas as pd

df = pd.read_csv('amtl.csv')
print(df[['feature3','feature4','feature5','feature6','feature7']].describe())
print('feature3 unique sample', df['feature3'].head(10).tolist())
