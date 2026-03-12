import pandas as pd

df = pd.read_csv('amtl.csv')
print(df[df['prob_male']==df['prob_male'].iloc[0]])
