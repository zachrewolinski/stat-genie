import pandas as pd

df=pd.read_csv('affairs.csv')

samples={
    'education':[547,1846,1575],
    'age':[4.962374447696927,-6.158222277261929,1.8405046718935036],
    'occupation':[52.0,27.0,47.0],
    'children':[4.0,7.0,10.0],
    'rating':[4,2,1],
    'yearsmarried':[18,14,20],
    'rownames':[7,6,3],
    'affairs':[5,1,3],
}
for col, vals in samples.items():
    present=[v for v in vals if v in set(df[col])]
    print(col, 'present', present)
