import pandas as pd


df = pd.read_csv('reading.csv')

# check variability of feature20 within participant
var_by_id = df.groupby('feature1')['feature20'].nunique()
print('feature20 nunique per feature1: min', var_by_id.min(), 'median', var_by_id.median(), 'max', var_by_id.max())
print('proportion with single value', (var_by_id==1).mean())

# check if feature4/5 etc vary within participant
for col in ['feature4','feature5','feature7','feature3']:
    nunique = df.groupby('feature1')[col].nunique()
    print(col, 'nunique per feature1 median', nunique.median(), 'max', nunique.max())

