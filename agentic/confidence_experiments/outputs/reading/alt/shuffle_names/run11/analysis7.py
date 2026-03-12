import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

wpm_from_running = df['num_words'] / df['running_time'] * 60
print(wpm_from_running.describe())
