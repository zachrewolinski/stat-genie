import pandas as pd

_df = pd.read_csv('amtl.csv')
_df = _df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'observable',
    'feature5': 'age',
    'feature6': 'age_uncert',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
})

req_cols = ['missing', 'observable', 'age', 'sex', 'tooth_class', 'genus']
df = _df.dropna(subset=req_cols).copy()

# Check invalids
print('rows', len(df))
print('observable <=0', (df['observable']<=0).sum())
print('missing <0', (df['missing']<0).sum())
print('missing > observable', (df['missing']>df['observable']).sum())
print('missing == observable', (df['missing']==df['observable']).sum())
print('missing/observable nan', ((df['missing']/df['observable']).isna()).sum())
print('missing/observable inf', ((df['missing']/df['observable']).replace([float('inf'), float('-inf')], pd.NA).isna()).sum())

# Show any problematic rows
bad = df[(df['observable']<=0) | (df['missing']<0) | (df['missing']>df['observable'])]
print('bad rows', len(bad))
if len(bad)>0:
    print(bad.head())

