import pandas as pd

df = pd.read_csv('reading.csv')
print(df['running_time'].describe())
