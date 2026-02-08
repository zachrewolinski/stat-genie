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

mask = (
    df['observable_sockets'].notna()
    & df['missing_teeth'].notna()
    & (df['observable_sockets'] > 0)
    & (df['missing_teeth'] >= 0)
    & (df['missing_teeth'] <= df['observable_sockets'])
    & df['age'].notna()
    & df['sex'].notna()
    & df['tooth_class'].notna()
    & df['genus'].notna()
)

df = df.loc[mask].copy()

prop = df['missing_teeth'] / df['observable_sockets']
print('rows', df.shape[0])
print('prop min', prop.min(), 'max', prop.max())
print('prop nan', prop.isna().sum())
print('weights min', df['observable_sockets'].min())

# any inf?
print('prop inf', np.isinf(prop).sum())
