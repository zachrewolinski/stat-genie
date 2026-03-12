import pandas as pd

df = pd.read_csv('hurricane.csv')
print(df[['gender_mf','year','ndam15']].corr())
