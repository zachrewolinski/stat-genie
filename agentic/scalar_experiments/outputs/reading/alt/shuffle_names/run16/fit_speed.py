import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

df = pd.read_csv('reading.csv')
num_words = df['num_words']

# compute wpm assuming adjusted time in ms
wpm_ms = num_words * 60000 / df['age']

X = wpm_ms.values.reshape(-1,1)
y = df['running_time'].values

model = LinearRegression().fit(X, y)
print('coef', model.coef_[0], 'intercept', model.intercept_, 'R2', model.score(X,y))

# compute wpm assuming adjusted time in centiseconds
wpm_cs = num_words * 6000 / df['age']
X2 = wpm_cs.values.reshape(-1,1)
model2 = LinearRegression().fit(X2, y)
print('cs coef', model2.coef_[0], 'intercept', model2.intercept_, 'R2', model2.score(X2,y))
