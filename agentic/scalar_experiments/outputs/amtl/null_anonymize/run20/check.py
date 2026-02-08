import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
cols = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing_teeth',
    'feature4': 'observable_sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
}

df = df.rename(columns=cols)
for col in ['missing_teeth','observable_sockets','age','sex']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

mask = df['observable_sockets'].notna() & df['missing_teeth'].notna()
print('total', df.shape[0])
print('valid counts', mask.sum())
print('obs <=0', (df['observable_sockets']<=0).sum())
print('missing <0', (df['missing_teeth']<0).sum())
print('missing > observable', (df['missing_teeth']>df['observable_sockets']).sum())
print('any nan age', df['age'].isna().sum())
print('any nan sex', df['sex'].isna().sum())
print('unique observable_sockets min', df['observable_sockets'].min())

mask2 = mask & (df['observable_sockets']>0) & (df['missing_teeth']>=0) & (df['missing_teeth']<=df['observable_sockets'])
print('mask2 rows', mask2.sum())
print('rows with missing_teeth == observable', (df['missing_teeth']==df['observable_sockets']).sum())
