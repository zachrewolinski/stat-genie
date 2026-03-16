import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

df = pd.read_csv('hurricane.csv')
print(df.head(10))
