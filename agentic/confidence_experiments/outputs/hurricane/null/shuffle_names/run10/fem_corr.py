import pandas as pd

df = pd.read_csv('hurricane.csv')
print(df[['category','ind','masfem_mturk']].corr())
