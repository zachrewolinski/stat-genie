import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
num_cols = df.select_dtypes(include='number').columns
corr = df[num_cols].corr()
print('Correlation of adjusted_running_time with num_words:', corr.loc['adjusted_running_time','num_words'])
print('Correlation of age with num_words:', corr.loc['age','num_words'])
print('Correlation of gender with num_words:', corr.loc['gender','num_words'])
