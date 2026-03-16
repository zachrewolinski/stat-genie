import pandas as pd
import numpy as np

df=pd.read_csv('reading.csv')

speed=df['running_time']
print('corr speed vs num_words', np.corrcoef(speed, df['num_words'])[0,1])
print('corr speed vs age(adjusted time)', np.corrcoef(speed, df['age'])[0,1])

