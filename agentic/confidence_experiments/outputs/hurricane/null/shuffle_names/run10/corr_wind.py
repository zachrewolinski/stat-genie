import pandas as pd

df = pd.read_csv('hurricane.csv')
num_cols = df.select_dtypes(include=['number']).columns
corr = df[num_cols].corr()['year'].sort_values(ascending=False)
print(corr)
