import pandas as pd

df = pd.read_csv('hurricane.csv')
for col in ['category','ind','masfem','year','wind']:
    print(col, df.groupby('masfem_mturk')[col].mean())
