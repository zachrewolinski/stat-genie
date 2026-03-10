import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

wpm_total = df['feature7'] / (df['feature4'] / 60000.0)

print('corr feature20 vs wpm_total', np.corrcoef(df['feature20'], wpm_total)[0,1])
print('wpm_total describe', wpm_total.describe())
print('median ratio feature20/wpm_total', (df['feature20']/wpm_total).median())

# check if feature20 equals (feature7/feature5)*some factor? compute best linear fit
from sklearn.linear_model import LinearRegression
X = wpm_total.values.reshape(-1,1)
y = df['feature20'].values
model = LinearRegression().fit(X,y)
print('linear fit for feature20 ~ wpm_total', model.coef_[0], model.intercept_, 'R2', model.score(X,y))

X2 = (df['feature7']/df['feature5']).values.reshape(-1,1)
model2 = LinearRegression().fit(X2,y)
print('linear fit for feature20 ~ feature7/feature5', model2.coef_[0], model2.intercept_, 'R2', model2.score(X2,y))

