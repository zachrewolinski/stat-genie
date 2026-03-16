import pandas as pd

df = pd.read_csv('amtl.csv')

spec = 'AMNH99.1/100'
print(df[df['prob_male']==spec][['sockets','age','genus','num_amtl','pop']])

spec = 'AMNH99.1/103'
print('\n', df[df['prob_male']==spec][['sockets','age','genus','num_amtl','pop']])
