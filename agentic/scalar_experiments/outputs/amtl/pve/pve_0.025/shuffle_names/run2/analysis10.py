import pandas as pd

df = pd.read_csv('amtl.csv')
print(df['tooth_class'].value_counts())
