import pandas as pd

df = pd.read_csv('reading.csv')
cols = ['adjusted_running_time','age','gender']
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        c1, c2 = cols[i], cols[j]
        print(c1, c2, df[c1].corr(df[c2]))
