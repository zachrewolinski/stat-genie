import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

pairs=[('adjusted_running_time','age','gender'),('age','adjusted_running_time','gender')]
for run, adj, scroll in pairs:
    diff = df[run] - (df[adj] + df[scroll])
    close = (diff.abs() < 5).mean()
    close50 = (diff.abs() < 50).mean()
    close100 = (diff.abs() < 100).mean()
    print(run, adj, scroll, 'close<5', close, 'close<50', close50, 'close<100', close100)
