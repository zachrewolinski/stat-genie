import pandas as pd

df = pd.read_csv('panda_nuts.csv')
clean = df[['age','sex','help','nuts_opened','seconds']].dropna()
clean = clean[clean['seconds']>0]
print(clean['sex'].value_counts())
print(clean['help'].value_counts())
