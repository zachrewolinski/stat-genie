import pandas as pd

df = pd.read_csv('hurricane.csv')
row = df[df['alldeaths']=='Katrina']
print(row)
