import pandas as pd

df = pd.read_csv('hurricane.csv')
print(df[['elapsedyrs','source']].corr())
print(df[['elapsedyrs','source']].describe())
