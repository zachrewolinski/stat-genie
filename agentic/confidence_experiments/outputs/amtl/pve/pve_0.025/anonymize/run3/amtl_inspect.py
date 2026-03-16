import pandas as pd
import numpy as np

csv_path = 'amtl.csv'

df = pd.read_csv(csv_path)
print(df.head())
print(df.dtypes)
print('rows', len(df))
# check if feature3 near integers
f3 = df['feature3']
nearest = np.round(f3)
print('feature3 near integer proportion (abs diff <1e-6):', np.mean(np.abs(f3-nearest)<1e-6))
print('feature3 near integer proportion (abs diff <0.1):', np.mean(np.abs(f3-nearest)<0.1))
print('feature3 min/max', f3.min(), f3.max())
# check if feature3 within [0, feature4]
print('feature3 within [0, feature4]:', np.mean((f3>=0) & (f3<=df["feature4"])))

# summary by genus
print(df.groupby('feature8')['feature3'].describe())
