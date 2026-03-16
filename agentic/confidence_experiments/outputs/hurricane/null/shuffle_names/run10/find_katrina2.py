import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv('hurricane.csv')
row = df[df['alldeaths']=='Katrina']
print(row)
