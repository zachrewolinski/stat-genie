import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df.tail())
print(df['education'].describe())
print('education unique sample', df['education'].sample(10, random_state=1).tolist())
print('education sorted head', np.sort(df['education'].unique())[:20])
print('education sorted tail', np.sort(df['education'].unique())[-20:])
