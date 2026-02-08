import pandas as pd
raw = pd.read_csv('amtl.csv')
viol = raw[raw['genus'] > raw['age']]
print(len(viol))
print(viol.head())
