import pandas as pd

df = pd.read_csv('mortgage.csv')
# gender col denied_PMI, approval deny
sub = df[['denied_PMI','deny']].dropna()
ct = pd.crosstab(sub['denied_PMI'], sub['deny'])
print(ct)
